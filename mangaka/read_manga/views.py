from django.shortcuts import render

def index(request):
    return render(request, 'read_manga/index.html')
