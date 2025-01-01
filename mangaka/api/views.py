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

# Définir le chemin des vidéos générées
def get_video_path(manga_id):
    return os.path.join(settings.MEDIA_ROOT, "manga_videos", f"{manga_id}")

def get_audio_duration_ffmpeg(audio_path):
    try:
        command = [
            "ffmpeg",
            "-i", audio_path,
            "-f", "null", "-",
        ]
        result = subprocess.run(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        stderr_output = result.stderr

        # Recherche de la ligne contenant la durée
        for line in stderr_output.splitlines():
            if "Duration" in line:
                duration_str = line.split("Duration:")[1].split(",")[0].strip()
                hours, minutes, seconds = map(float, duration_str.split(":"))
                return hours * 3600 + minutes * 60 + seconds

        raise ValueError("Could not find duration in ffmpeg output")
    except Exception as e:
        print(f"Error extracting duration: {e}")
        return 1.0



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
                audio_duration = get_audio_duration_ffmpeg(panel.audio_file.path)
            else:
                audio_duration = 2


            if panel.audio_file:
                ffmpeg_command = [
                    "ffmpeg", "-y",
                    "-loop", "1",
                    "-i", image_path,
                    "-i", panel.audio_file.path,
                    "-vf", (
                        "scale='if(gt(a,16/9),854,-2):if(gt(a,16/9),-2,480)',"
                        "pad=854:480:(ow-iw)/2:(oh-ih)/2:white"
                    ),
                    "-pix_fmt", "yuv420p",
                    "-c:v", "libx264",
                    "-c:a", "aac", "-ar", "22050", "-b:a", "128k",
                    "-t", str(audio_duration),
                    "-shortest",
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
                    "-shortest",
                    temp_video_path
                ]

            subprocess.run(ffmpeg_command, check=True)

        # Créer un fichier de concaténation
        concat_file_path = os.path.join(video_folder, "concat_list.txt")
        with open(concat_file_path, "w") as concat_file:
            for temp_video in temp_videos:
                concat_file.write(f"file '{temp_video}'\n")

        # Générer la vidéo finale
        concatenated_video_path = os.path.join(video_folder, f"{manga_id}.mp4")
        final_ffmpeg_command = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file_path,
            "-i", background_audio_path,
            "-filter_complex", "[1:a]volume=0.25[audio2];[0:a][audio2]amix=inputs=2:duration=shortest",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-ar", "22050", "-b:a", "128k",
            "-pix_fmt", "yuv420p",
            concatenated_video_path
        ]
        subprocess.run(final_ffmpeg_command, check=True)

        # Nettoyer les fichiers temporaires
        for temp_video in temp_videos:
            os.remove(temp_video)
        os.remove(concat_file_path)

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