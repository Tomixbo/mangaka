# Project Description
Mangaka is a web application designed to enhance the manga reading experience by providing a smoother and more intuitive way to enjoy your favorite stories. Instead of navigating manually through manga pages, Mangaka automatically detects and extracts individual panels from each page, displaying them one at a time for easier reading.

In addition, Mangaka offers a hands-free experience: simply press play, and the application will sequentially display the panels while narrating the story for you. This eliminates the need for constant manual interaction, allowing you to fully immerse yourself in the manga's plot.

With Mangaka, following your favorite manga has never been this effortless or enjoyable!

# Environment and dependencies installation
```
python -m venv .venv
.venv\Scripts\activate
.venv\Scripts\python -m pip install -r requirements.txt
cd mangaka\theme\static_src
python manage.py tailwind install
```

# Deployment in development
bash 01 : 
```
.venv\Scripts\activate
cd mangaka
python manage.py tailwind start
```
bash 02 :
```
.venv\Scripts\activate
cd mangaka
python manage.py runserver

```

# Demo
[![Video Demo](https://img.youtube.com/vi/GvON2_ZyZ_4/0.jpg)](https://www.youtube.com/watch?v=GvON2_ZyZ_4)
