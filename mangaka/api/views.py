import os
import subprocess
from django.http import JsonResponse, FileResponse, Http404
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from read_manga.models import Manga  # Import du modèle Manga
import uuid
from django.shortcuts import get_object_or_404
from django.http import HttpResponseRedirect
from read_manga.models import Manga, Panel
import json
import cv2
import numpy as np
import wave
from moviepy.editor import VideoFileClip
from moviepy.video.fx.all import resize, crop
import shutil



def convert_to_wav(input_path, output_path=None):

    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}_pcm.wav"

    try:
        # Commande FFmpeg pour convertir en WAV PCM non compressé
        command = [
            "ffmpeg", "-y",  # Écraser les fichiers existants
            "-i", input_path,  # Fichier d'entrée
            "-ar", "44100",  # Fréquence d'échantillonnage
            "-ac", "2",  # Nombre de canaux (stéréo)
            "-c:a", "pcm_s16le",  # Codec audio (PCM 16 bits)
            output_path  # Fichier de sortie
        ]

        # Exécuter la commande
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_path

    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de la conversion : {e.stderr.decode()}")
        raise Exception("Échec de la conversion du fichier audio en WAV non compressé.")

def get_audio_duration(file_path):
   with wave.open(file_path, 'r') as audio_file:
      frame_rate = audio_file.getframerate()
      n_frames = audio_file.getnframes()
      duration = n_frames / float(frame_rate)
      return round(duration, 3)

# Définir le chemin des vidéos générées
def get_video_path(manga_id):
    return os.path.join(settings.MEDIA_ROOT, "manga_videos", f"{manga_id}", f"{manga_id}.mp4")

def filter_image(image_path, temp_directory, panel_id):
    """
    Applies effects (sharpening, smoothing, and brightness adjustment) to the manga image and saves it temporarily.

    Args:
        image_path (str): Path to the input image.
        temp_directory (str): Directory to save the temporary processed image.
        panel_id (int): Unique ID for the panel (used to name the file).

    Returns:
        str: Path to the processed image file.
    """
    # Read the input image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to open image file: {image_path}")

    # Apply smoothing (blur)
    blur_value = 1
    smoothed_image = cv2.GaussianBlur(image, (2 * blur_value + 1, 2 * blur_value + 1), 0)

    # Apply sharpening
    sharpening_kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
    sharpened_image = cv2.filter2D(smoothed_image, -1, sharpening_kernel)

    # Adjust brightness (optional)
    brightness = 10  # Adjust brightness (positive to brighten, negative to darken)
    processed_image = cv2.convertScaleAbs(sharpened_image, alpha=1.0, beta=brightness)

    # Save the processed image to the temporary directory
    temp_image_path = os.path.join(temp_directory, f"processed_panel_{panel_id}.jpg")
    cv2.imwrite(temp_image_path, processed_image)

    return temp_image_path



def smooth_speed_curve(t, accel_duration=0.5, total_duration=1.0):
    """
    Generates a smooth speed curve with acceleration at the beginning and end.
    """
    accel_fraction = accel_duration / total_duration  # Fraction of time for acceleration
    middle_start = accel_fraction
    middle_end = 1.0 - accel_fraction

    if t < middle_start:  # Accelerating phase at the start
        return 1
    elif t > middle_end:  # Accelerating phase at the end
        return 1
    else:  # Constant-speed phase in the middle
        return 0.1

def apply_zoom_to_video(input_video_path, output_video_path, start_scale=1.0, end_scale=1.1, fps=30):

    # Determine the root directory of the input video
    root_dir = os.path.dirname(input_video_path)
    temp_audio_path = os.path.join(root_dir, "temp_audio.aac")
    temp_video_path = os.path.join(root_dir, "temp_video.mp4")

    # Step 1: Extract Audio
    try:
        extract_audio_command = [
            "ffmpeg", "-i", input_video_path, "-vn",  # Extract audio only
            "-acodec", "copy", temp_audio_path, "-y"  # Overwrite if exists
        ]
        subprocess.run(extract_audio_command, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        audio_present = True
    except subprocess.CalledProcessError:
        print("No audio found in the input video.")
        audio_present = False

    # Step 2: Apply Zoom and Fade (Video Processing)
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {input_video_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    # Prepare the video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    # Precompute scale factors
    times = np.linspace(0, 1, total_frames)  # Normalized time (0 to 1)
    speed_factors = np.array([smooth_speed_curve(t, accel_duration=0.5, total_duration=duration) for t in times])

    # Normalize speed factors to ensure monotonic progression
    cumulative_speed = np.cumsum(speed_factors)
    cumulative_speed /= cumulative_speed[-1]  # Normalize to 1.0 at the end

    # Map normalized cumulative speed to scales
    scales = start_scale + (end_scale - start_scale) * cumulative_speed

    # Center of the frame
    center_x, center_y = width // 2, height // 2
    fade_duration = int(0.5 * fps)  # Number of frames for fade-in/out

    for frame_index in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Get the current zoom scale
        scale = scales[frame_index]

        # Compute the transformation matrix
        transform_matrix = cv2.getRotationMatrix2D((center_x, center_y), 0, scale)

        # Apply the transformation
        transformed_frame = cv2.warpAffine(
            frame,
            transform_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderValue=(255, 255, 255),  # White padding
        )

        # Apply fade-in effect
        if frame_index < fade_duration:
            alpha = frame_index / fade_duration
            white_frame = np.ones_like(transformed_frame) * 255
            transformed_frame = cv2.addWeighted(transformed_frame, alpha, white_frame, 1 - alpha, 0)

        # Apply fade-out effect
        if frame_index >= total_frames - fade_duration:
            alpha = (total_frames - frame_index + 1) / fade_duration
            white_frame = np.ones_like(transformed_frame) * 255
            transformed_frame = cv2.addWeighted(transformed_frame, alpha, white_frame, 1 - alpha, 0)

            # Forcer à blanc pour la dernière frame
            if frame_index >= total_frames - 2:
                transformed_frame = white_frame

        # Write the transformed frame
        out.write(transformed_frame)

    cap.release()
    out.release()

    # Step 3: Merge Audio Back (if audio exists)
    if audio_present:
        merge_command = [
            "ffmpeg", "-i", temp_video_path, "-i", temp_audio_path,
            "-c:v", "copy", "-c:a", "aac", "-strict", "experimental",
            output_video_path, "-y"
        ]
    else:
        merge_command = [
            "ffmpeg", "-i", temp_video_path,
            "-c:v", "copy", "-an", output_video_path, "-y"  # No audio
        ]
    subprocess.run(merge_command, check=True)

    # Step 4: Clean up temporary files
    try:
        os.remove(temp_video_path)
        if audio_present:
            os.remove(temp_audio_path)
    except OSError as e:
        print(f"Error removing temporary files: {e}")


@csrf_exempt
@require_POST
def generate_video_api(request, manga_id):
    manga = get_object_or_404(Manga, id=manga_id)
    video_folder = os.path.join(settings.MEDIA_ROOT, "manga_videos", f"{manga_id}")
    os.makedirs(video_folder, exist_ok=True)
    background_audio_path = os.path.join(settings.MEDIA_ROOT, "background_music", "background_music.mp3")  

    # Nettoyer les fichiers temporaires
    temp_videos = []
    try:
        # Récupérer les panneaux triés
        pages = manga.pages.order_by('id')
        panels = Panel.objects.filter(manga_page__in=pages).order_by('manga_page_id', 'order')

        if not panels.exists():
            return JsonResponse({"success": False, "message": "No panels found for the manga."})
        
        temp_image_directory = os.path.join(video_folder, "temp_images")
        os.makedirs(temp_image_directory, exist_ok=True)
        # Générer une vidéo pour chaque panel
        for panel in panels:
            
            # Process the image and get the temporary path
            image_path = filter_image(panel.image.path, temp_image_directory, panel.id)
            temp_video_path = os.path.join(video_folder, f"temp_panel_{panel.id}.mp4")
            temp_videos.append(temp_video_path)

            if panel.audio_file:
                audio_path = convert_to_wav(panel.audio_file.path, output_path=os.path.join(video_folder, f"{panel.id}_pcm.wav"))
                audio_duration = get_audio_duration(audio_path)
                os.remove(audio_path)
                
            else:
                audio_duration = 2

            if panel.audio_file:
                ffmpeg_command = [
                    "ffmpeg", "-y",
                    "-loop", "1",
                    "-i", image_path,
                    "-i", panel.audio_file.path,
                    "-filter_complex", (
                        f"[1:a]apad=pad_dur={audio_duration+0.5},aresample=async=1:min_hard_comp=0.100:first_pts=0[audio];"
                        "[0:v]scale='if(gt(a,16/9),1920*0.92,-2):if(gt(a,16/9),-2,1080*0.92)',"
                        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:white[scaled];"
                        "[scaled][audio]concat=n=1:v=1:a=1[outv][outa]"
                    ),
                    "-map", "[outv]",
                    "-map", "[outa]",
                    "-pix_fmt", "yuv420p",
                    "-c:v", "libx264",
                    "-c:a", "aac", "-ar", "22050", "-b:a", "128k",
                    "-t", str(audio_duration+0.5),
                    temp_video_path
                ]
            else:
                ffmpeg_command = [
                    "ffmpeg", "-y",
                    "-loop", "1",
                    "-i", image_path,
                    "-f", "lavfi", "-t", "2.5", "-i", "anullsrc=r=22050:cl=mono",
                    "-vf", (
                        "scale='if(gt(a,16/9),1920*0.92,-2):if(gt(a,16/9),-2,1080*0.92)',"
                        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:white"
                    ),
                    "-pix_fmt", "yuv420p",
                    "-c:v", "libx264",
                    "-c:a", "aac", "-ar", "22050", "-b:a", "128k",
                    "-t", "2.5",
                    temp_video_path
                ]

            subprocess.run(ffmpeg_command, check=True)

            # Step 2: Apply zoom effect using OpenCV
            zoomed_video_path = os.path.join(video_folder, f"zoomed_panel_{panel.id}.mp4")
            apply_zoom_to_video(temp_video_path, zoomed_video_path, start_scale=1.0, end_scale=1.1, fps=30)

            # Replace temp_video_path with zoomed_video_path for concatenation
            temp_videos[-1] = zoomed_video_path
            os.remove(temp_video_path)  # Clean up intermediate video
        
        shutil.rmtree(temp_image_directory)
        # Créer un fichier de concaténation
        concat_file_path = os.path.join(video_folder, "concat_list.txt")
        with open(concat_file_path, "w") as concat_file:
            for temp_video in temp_videos:
                concat_file.write(f"file '{temp_video}'\n")

        # Générer la vidéo finale
        concatenated_video_path = os.path.join(video_folder, f"{manga_id}_concat.mp4")
        sync_video_path = os.path.join(video_folder, f"{manga_id}_sync.mp4")
        final_video_path = os.path.join(video_folder, f"{manga_id}.mp4")
        # Étape 1 : Générer la vidéo concaténée
        concat_ffmpeg_command = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-ar", "22050", "-b:a", "128k",
            "-pix_fmt", "yuv420p",
            concatenated_video_path
        ]
        subprocess.run(concat_ffmpeg_command, check=True)

        # Synchronisation finale après la concaténation
        sync_command = [
            "ffmpeg", "-y",
            "-i", concatenated_video_path,
            "-vf", "setpts=PTS-STARTPTS",
            "-af", "aresample=async=1",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-ar", "22050", "-b:a", "128k",
            sync_video_path
        ]
        subprocess.run(sync_command, check=True)


        # Étape 2 : Obtenir la durée totale de la vidéo concaténée
        final_ffmpeg_command = [
            "ffmpeg", "-y",
            "-i", sync_video_path,  # Vidéo concaténée
            "-stream_loop", "-1", "-i", background_audio_path,  # Audio de fond en boucle
            "-filter_complex", (
                "[0:a]apad,aresample=async=1[audio_base];"  # Ajout de silences si nécessaire
                "[1:a]volume=0.25[audio_bg];"  # Réduction du volume de l’audio de fond
                "[audio_base][audio_bg]amix=inputs=2:duration=longest[audio_mix]"  # Combinaison des audios
            ),
            "-map", "0:v",  # Vidéo principale
            "-map", "[audio_mix]",  # Audio mixé
            "-shortest",  # Arrêter dès que la vidéo est terminée
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-ar", "22050", "-b:a", "128k",
            "-pix_fmt", "yuv420p",
            final_video_path
        ]
        subprocess.run(final_ffmpeg_command, check=True)



        # Nettoyer les fichiers temporaires
        for temp_video in temp_videos:
            os.remove(temp_video)
        os.remove(concat_file_path)
        os.remove(sync_video_path)
        os.remove(concatenated_video_path)
        

        return JsonResponse({"success": True, "message": "Video generated successfully."})

    except subprocess.CalledProcessError as e:
        return JsonResponse({"success": False, "message": f"FFmpeg error: {str(e)}"}, status=500)
    except Exception as e:
        return JsonResponse({"success": False, "message": f"Unexpected error: {str(e)}"}, status=500)




    
def check_video_exists(request, manga_id):
    """
    Endpoint pour vérifier si une vidéo existe pour un manga donné.
    """
    try:
        # Chemin de la vidéo générée
        video_path = get_video_path(manga_id)

        # Vérifier si le fichier existe
        if os.path.exists(video_path):
            return JsonResponse({"success": True, "exists": True, "message": "Vidéo disponible."})
        else:
            return JsonResponse({"success": True, "exists": False, "message": "Vidéo non disponible."})
    except Exception as e:
        return JsonResponse({"success": False, "message": f"Erreur inattendue : {str(e)}"}, status=500)


def download_video(request, manga_id):
    file_path = os.path.join(settings.MEDIA_ROOT, "manga_videos", f"{manga_id}", f"{manga_id}.mp4")
    if not os.path.exists(file_path):
        return JsonResponse({"error": "File not found"}, status=404)
    response = FileResponse(open(file_path, 'rb'))
    response['Content-Disposition'] = 'attachment; filename="manga_reader_video.mp4"'
    return response