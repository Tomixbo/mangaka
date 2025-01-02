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
import wave




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

        # Générer une vidéo pour chaque panel
        for panel in panels:
            image_path = panel.image.path
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
                        f"[1:a]apad=pad_dur={audio_duration},aresample=async=1:min_hard_comp=0.100:first_pts=0[audio];"
                        "[0:v]scale='if(gt(a,16/9),854,-2):if(gt(a,16/9),-2,480)',"
                        "pad=854:480:(ow-iw)/2:(oh-ih)/2:white[scaled];"
                        "[scaled][audio]concat=n=1:v=1:a=1[outv][outa]"
                    ),
                    "-map", "[outv]",
                    "-map", "[outa]",
                    "-pix_fmt", "yuv420p",
                    "-c:v", "libx264",
                    "-c:a", "aac", "-ar", "22050", "-b:a", "128k",
                    "-t", str(audio_duration),
                    temp_video_path
                ]
            else:
                ffmpeg_command = [
                    "ffmpeg", "-y",
                    "-loop", "1",
                    "-i", image_path,
                    "-f", "lavfi", "-t", "2", "-i", "anullsrc=r=22050:cl=mono",
                    "-vf", (
                        "scale='if(gt(a,16/9),854,-2):if(gt(a,16/9),-2,480)',"
                        "pad=854:480:(ow-iw)/2:(oh-ih)/2:white"
                    ),
                    "-pix_fmt", "yuv420p",
                    "-c:v", "libx264",
                    "-c:a", "aac", "-ar", "22050", "-b:a", "128k",
                    "-t", "2",
                    temp_video_path
                ]

            subprocess.run(ffmpeg_command, check=True)

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