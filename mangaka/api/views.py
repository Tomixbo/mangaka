import os
import subprocess
from django.http import JsonResponse, FileResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.shortcuts import get_object_or_404
import concurrent.futures
import cv2
import numpy as np
import wave
import shutil

from read_manga.models import Manga, Panel


def convert_to_wav(input_path, output_path=None):
    """
    Convertit un fichier audio en WAV PCM non compressé (44100Hz, stéréo, 16 bits).
    """
    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}_pcm.wav"

    try:
        command = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ar", "44100",
            "-ac", "2",
            "-c:a", "pcm_s16le",
            output_path
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de la conversion : {e.stderr.decode()}")
        raise Exception("Échec de la conversion du fichier audio en WAV non compressé.")


def get_audio_duration(file_path):
    """
    Retourne la durée (en secondes, arrondie) d'un fichier WAV.
    """
    with wave.open(file_path, 'r') as audio_file:
        frame_rate = audio_file.getframerate()
        n_frames = audio_file.getnframes()
        duration = n_frames / float(frame_rate)
        return round(duration, 3)


def get_video_path(manga_id):
    """
    Chemin de la vidéo finale pour un manga donné.
    """
    return os.path.join(settings.MEDIA_ROOT, "manga_videos", f"{manga_id}", f"{manga_id}.mp4")


def filter_image(image_path, temp_directory, panel_id):
    """
    Applique un lissage (blur), un sharpen, et ajuste la luminosité sur l'image,
    puis la sauvegarde dans un fichier temporaire unique.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to open image file: {image_path}")

    # Applique un flou léger
    blur_value = 1
    smoothed_image = cv2.GaussianBlur(image, (2 * blur_value + 1, 2 * blur_value + 1), 0)

    # Sharpen
    sharpening_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    sharpened_image = cv2.filter2D(smoothed_image, -1, sharpening_kernel)

    # Ajuste la luminosité (beta = +10 = un peu plus clair)
    brightness = 10
    processed_image = cv2.convertScaleAbs(sharpened_image, alpha=1.0, beta=brightness)

    # Chemin unique pour l'image temporaire
    temp_image_path = os.path.join(temp_directory, f"processed_panel_{panel_id}.jpg")
    cv2.imwrite(temp_image_path, processed_image)
    return temp_image_path


def smooth_speed_curve(t, accel_duration=0.5, total_duration=1.0):
    """
    Génère une courbe de vitesse lissée (accélération en début et fin).
    """
    accel_fraction = accel_duration / total_duration
    middle_start = accel_fraction
    middle_end = 1.0 - accel_fraction

    if t < middle_start:
        return 1
    elif t > middle_end:
        return 1
    else:
        return 0.1


def apply_zoom_to_video(
    input_video_path,
    output_video_path,
    panel_id,
    start_scale=1.0,
    end_scale=1.1,
    fps=30
):
    """
    Lit la vidéo `input_video_path` via OpenCV, applique un zoom progressif,
    un fade-in/out, et réinsère l'audio si présent, le tout dans des fichiers
    temporaires uniques (pour éviter toute collision en parallèle).
    """
    # Répertoire racine du fichier d'entrée
    root_dir = os.path.dirname(input_video_path)

    # Noms de fichiers temporaires uniques
    temp_audio_path = os.path.join(root_dir, f"temp_audio_{panel_id}.aac")
    temp_video_path = os.path.join(root_dir, f"temp_video_{panel_id}.mp4")

    # 1) Extraction de l'audio s'il y en a
    audio_present = False
    try:
        extract_audio_cmd = [
            "ffmpeg", "-y",
            "-i", input_video_path,
            "-vn",  # pas de flux vidéo, juste extraire l'audio
            "-acodec", "copy",
            temp_audio_path
        ]
        subprocess.run(extract_audio_cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        audio_present = True
    except subprocess.CalledProcessError:
        print(f"Aucun flux audio pour panel {panel_id} (ce n'est pas forcément une erreur).")

    # 2) Application du zoom & fade via OpenCV
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {input_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Pour éviter division par zéro si la vidéo est vide
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video file has no frames: {input_video_path}")

    duration = total_frames / fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    # Pré-calcul des facteurs de zoom
    times = np.linspace(0, 1, total_frames)  # fraction de 0 à 1
    speed_factors = np.array([
        smooth_speed_curve(t, accel_duration=0.5, total_duration=duration)
        for t in times
    ])
    # On cumule la "vitesse" pour obtenir un zoom progressif
    cumulative_speed = np.cumsum(speed_factors)
    if cumulative_speed[-1] == 0:
        # vidéo bizarrement sans progression => fallback
        cumulative_speed = np.linspace(0, 1, total_frames)
    else:
        cumulative_speed /= cumulative_speed[-1]

    scales = start_scale + (end_scale - start_scale) * cumulative_speed

    center_x, center_y = width // 2, height // 2
    fade_duration = int(0.5 * fps)

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        scale = scales[frame_index]
        transform_matrix = cv2.getRotationMatrix2D((center_x, center_y), 0, scale)

        transformed_frame = cv2.warpAffine(
            frame,
            transform_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderValue=(255, 255, 255),  # fond blanc
        )

        # Fade-in (début)
        if frame_index < fade_duration:
            alpha = frame_index / fade_duration
            white_frame = np.full_like(transformed_frame, 255)
            transformed_frame = cv2.addWeighted(transformed_frame, alpha, white_frame, 1 - alpha, 0)

        # Fade-out (fin)
        if frame_index >= total_frames - fade_duration:
            alpha = (total_frames - frame_index) / fade_duration
            alpha = max(0, min(1, alpha))
            white_frame = np.full_like(transformed_frame, 255)
            transformed_frame = cv2.addWeighted(transformed_frame, alpha, white_frame, 1 - alpha, 0)

            # Forcer à blanc pour la dernière frame
            if frame_index >= total_frames - 2:
                transformed_frame = white_frame

        out.write(transformed_frame)
        frame_index += 1

    cap.release()
    out.release()

    # 3) Réintégration de l'audio (si présent)
    if audio_present:
        merge_cmd = [
            "ffmpeg", "-y",
            "-i", temp_video_path,
            "-i", temp_audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-strict", "experimental",
            output_video_path
        ]
        subprocess.run(merge_cmd, check=True)
    else:
        # Pas d'audio
        noaudio_cmd = [
            "ffmpeg", "-y",
            "-i", temp_video_path,
            "-c:v", "copy",
            "-an",
            output_video_path
        ]
        subprocess.run(noaudio_cmd, check=True)

    # 4) Nettoyage
    try:
        os.remove(temp_video_path)
        if audio_present:
            os.remove(temp_audio_path)
    except OSError as e:
        print(f"Error removing temporary files for panel {panel_id}: {e}")


def create_video_for_panel(panel, video_folder, temp_image_directory):
    """
    Enchaîne toutes les étapes pour générer la vidéo d'un panel :
      1) Filtrage de l'image
      2) Génération d'une courte vidéo (image + audio)
      3) Application du zoom/fade via OpenCV
    Renvoie le chemin final de la vidéo (zoomée).
    """
    # 1) Filtrage de l'image
    processed_image_path = filter_image(panel.image.path, temp_image_directory, panel.id)

    # 2) Construction du panel en vidéo (temp_panel_{panel.id}.mp4)
    temp_video_path = os.path.join(video_folder, f"temp_panel_{panel.id}.mp4")
    zoomed_video_path = os.path.join(video_folder, f"zoomed_panel_{panel.id}.mp4")

    # Calcul de la durée audio si panel.audio_file
    audio_duration = 2
    if panel.audio_file:
        wav_path = os.path.join(video_folder, f"{panel.id}_pcm.wav")
        convert_to_wav(panel.audio_file.path, output_path=wav_path)
        audio_duration = get_audio_duration(wav_path)
        try:
            os.remove(wav_path)
        except OSError:
            pass

    # FFmpeg pour générer la vidéo initiale
    if panel.audio_file:
        # Avec audio
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", processed_image_path,
            "-i", panel.audio_file.path,
            "-filter_complex", (
                f"[1:a]apad=pad_dur={audio_duration+0.5},aresample=async=1:"
                "min_hard_comp=0.100:first_pts=0[audio];"
                "[0:v]scale='if(gt(a,16/9),1920*0.92,-2):if(gt(a,16/9),-2,1080*0.92)',"
                "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:white[scaled];"
                "[scaled][audio]concat=n=1:v=1:a=1[outv][outa]"
            ),
            "-map", "[outv]",
            "-map", "[outa]",
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264",
            "-c:a", "aac", "-ar", "22050", "-b:a", "128k",
            "-t", str(audio_duration + 0.5),
            temp_video_path
        ]
    else:
        # Sans audio => 2.5s de silence
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", processed_image_path,
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

    subprocess.run(ffmpeg_cmd, check=True)

    # 3) Appliquer le zoom/fade
    apply_zoom_to_video(
        input_video_path=temp_video_path,
        output_video_path=zoomed_video_path,
        panel_id=panel.id,  # pour des noms de fichiers uniques
        start_scale=1.0,
        end_scale=1.1,
        fps=30
    )

    # Nettoyer la vidéo intermédiaire
    try:
        os.remove(temp_video_path)
    except OSError:
        pass

    return zoomed_video_path

def get_ffmpeg_video_args(use_gpu=False):
    """
    Retourne la liste d'arguments vidéo pour FFmpeg en fonction
    de la disponibilité ou non du GPU (CUDA/NVENC).
    """
    if use_gpu:
        # Accélération GPU (NVENC)
        return [
            "-c:v", "h264_nvenc",
            "-preset", "fast",
            "-b:v", "5M",
        ]
    else:
        # Encodage logiciel (libx264 + CRF)
        return [
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
        ]

def is_cuda_available():
    """
    Retourne True si FFmpeg trouve l'accélération CUDA, False sinon.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-hwaccels"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        output = result.stdout.lower() + result.stderr.lower()
        return "cuda" in output
    except subprocess.CalledProcessError:
        # Si la commande -hwaccels échoue, on considère que CUDA n'est pas dispo
        return False


@csrf_exempt
@require_POST
def generate_video_api(request, manga_id):
    """
    Génère la vidéo finale pour un manga :
      - Création en parallèle des vidéos panel par panel
      - Concaténation
      - Ajout d'une musique de fond
    """
    manga = get_object_or_404(Manga, id=manga_id)
    video_folder = os.path.join(settings.MEDIA_ROOT, "manga_videos", f"{manga_id}")
    os.makedirs(video_folder, exist_ok=True)

    background_audio_path = os.path.join(settings.MEDIA_ROOT, "background_music", "background_music.mp3")

    # Répertoire pour les images intermédiaires
    temp_image_directory = os.path.join(video_folder, "temp_images")
    os.makedirs(temp_image_directory, exist_ok=True)

    try:
        panels = Panel.objects.filter(manga_page__manga=manga).order_by('manga_page_id', 'order')
        if not panels.exists():
            return JsonResponse({"success": False, "message": "No panels found for the manga."})

        # On crée les vidéos en parallèle
        temp_videos_dict = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            future_to_panel_id = {}
            for panel in panels:
                future = executor.submit(create_video_for_panel, panel, video_folder, temp_image_directory)
                future_to_panel_id[future] = panel.id

            for future in concurrent.futures.as_completed(future_to_panel_id):
                panel_id = future_to_panel_id[future]
                try:
                    zoomed_video_path = future.result()
                    temp_videos_dict[panel_id] = zoomed_video_path
                except Exception as e:
                    return JsonResponse({
                        "success": False,
                        "message": f"Error generating video for panel {panel_id}: {str(e)}"
                    }, status=500)

        # Reconstituer la liste des vidéos dans l'ordre
        temp_videos = []
        for panel in panels:
            temp_videos.append(temp_videos_dict[panel.id])

        # Nettoyer le dossier d'images
        shutil.rmtree(temp_image_directory)

        # Fichier de concat
        concat_file_path = os.path.join(video_folder, "concat_list.txt")
        with open(concat_file_path, "w") as f:
            for tv_path in temp_videos:
                f.write(f"file '{tv_path}'\n")

        # Chemins finaux
        concatenated_video_path = os.path.join(video_folder, f"{manga_id}_concat.mp4")
        sync_video_path = os.path.join(video_folder, f"{manga_id}_sync.mp4")
        final_video_path = os.path.join(video_folder, f"{manga_id}.mp4")
        gpu_available = is_cuda_available()
        # On récupère les arguments vidéo selon qu'on a le GPU ou non
        video_args_concat = get_ffmpeg_video_args(use_gpu=gpu_available)
        video_args_sync   = get_ffmpeg_video_args(use_gpu=gpu_available)
        video_args_final  = get_ffmpeg_video_args(use_gpu=gpu_available)

        #------------------------------------------
        # 1) Concaténer
        #------------------------------------------
        concat_cmd = [
            "ffmpeg", 
            "-y",
        ]
        # Si on veut décoder (ou au moins préparer) via GPU, on met -hwaccel cuda ici,
        # AVANT le -i concat_file.txt
        if gpu_available:
            concat_cmd += ["-hwaccel", "cuda"]

        concat_cmd += [
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file_path,
            # Ajout des args d'encodage vidéo
            *video_args_concat,
            # Audio commun
            "-c:a", "aac", "-ar", "22050", "-b:a", "128k",
            "-pix_fmt", "yuv420p",
            concatenated_video_path
        ]
        subprocess.run(concat_cmd, check=True)

        #------------------------------------------
        # 2) Synchronisation
        #------------------------------------------
        sync_cmd = [
            "ffmpeg",
            "-y",
        ]
        if gpu_available:
            sync_cmd += ["-hwaccel", "cuda"]

        sync_cmd += [
            "-i", concatenated_video_path,
            "-vf", "setpts=PTS-STARTPTS",
            "-af", "aresample=async=1",
            *video_args_sync,
            "-c:a", "aac", "-ar", "22050", "-b:a", "128k",
            sync_video_path
        ]
        subprocess.run(sync_cmd, check=True)

        #------------------------------------------
        # 3) Musique de fond
        #------------------------------------------
        final_cmd = [
            "ffmpeg",
            "-y",
        ]
        if gpu_available:
            final_cmd += ["-hwaccel", "cuda"]

        # Ici, on a DEUX entrées : la vidéo sync_video_path et l'audio background_audio_path.
        # L'accélération GPU pour la première entrée (vidéo) est utile, la seconde est seulement audio.
        final_cmd += [
            "-i", sync_video_path,
            "-stream_loop", "-1", 
            "-i", background_audio_path,
            "-filter_complex", (
                "[0:a]apad,aresample=async=1[audio_base];"
                "[1:a]volume=0.25[audio_bg];"
                "[audio_base][audio_bg]amix=inputs=2:duration=longest[audio_mix]"
            ),
            "-map", "0:v",
            "-map", "[audio_mix]",
            "-shortest",
            *video_args_final,
            "-c:a", "aac", "-ar", "22050", "-b:a", "128k",
            "-pix_fmt", "yuv420p",
            final_video_path
        ]
        subprocess.run(final_cmd, check=True)

        # Nettoyage
        for tv_path in temp_videos:
            os.remove(tv_path)
        os.remove(concat_file_path)
        os.remove(sync_video_path)
        os.remove(concatenated_video_path)

        return JsonResponse({"success": True, "message": "Video generated successfully."})

    except subprocess.CalledProcessError as e:
        return JsonResponse({
            "success": False,
            "message": f"FFmpeg error: {str(e)}"
        }, status=500)
    except Exception as e:
        return JsonResponse({
            "success": False,
            "message": f"Unexpected error: {str(e)}"
        }, status=500)


def check_video_exists(request, manga_id):
    """
    Vérifie si la vidéo finale d'un manga existe déjà.
    """
    try:
        video_path = get_video_path(manga_id)
        if os.path.exists(video_path):
            return JsonResponse({"success": True, "exists": True, "message": "Vidéo disponible."})
        else:
            return JsonResponse({"success": True, "exists": False, "message": "Vidéo non disponible."})
    except Exception as e:
        return JsonResponse({"success": False, "message": f"Erreur inattendue : {str(e)}"}, status=500)


def download_video(request, manga_id):
    """
    Permet de télécharger la vidéo finale d'un manga, si elle existe.
    """
    file_path = os.path.join(settings.MEDIA_ROOT, "manga_videos", f"{manga_id}", f"{manga_id}.mp4")
    if not os.path.exists(file_path):
        return JsonResponse({"error": "File not found"}, status=404)
    response = FileResponse(open(file_path, 'rb'))
    response['Content-Disposition'] = 'attachment; filename="manga_reader_video.mp4"'
    return response
