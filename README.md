# Emotion Detector using Webcam 🎭

## Overview

This project is a **real-time emotion detection system** that uses a webcam to detect human facial expressions and classify emotions such as **Happy, Sad, Angry, Neutral, Surprise, and Fear**. It uses a **pre-trained deep learning model (.h5)** along with **OpenCV** for face detection.

## Features

* Real-time emotion detection using webcam
* Face detection using Haar Cascade classifier
* Deep learning model for emotion classification
* Lightweight and easy to run

## Project Structure

```
emotion_labels.py              # Contains emotion label names
emotion_model.h5               # Trained emotion detection model
haarcascade_frontalface_default.xml   # Face detection classifier
webcam_emotion.py              # Main script to run emotion detection
requirements.txt               # Required Python libraries
```

## Requirements

Install dependencies using:

```
pip install -r requirements.txt
```

## How to Run

```
python webcam_emotion.py
```


Developed for AI/ML emotion recognition project.
