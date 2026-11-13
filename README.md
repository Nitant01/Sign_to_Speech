# SignSpeak
### Static Sign Language to Speech Translation

SignSpeak recognizes static American Sign Language (ASL) alphabet gestures using a webcam and converts recognized gestures into text and speech.

## Features
- Real-time webcam hand detection
- Static ASL alphabet recognition (A-Z)
- Text construction with Space and Delete controls
- Text-to-speech with Male and Female voices
- Adjustable speech speed and recognition settings

## Technology Stack
- Python 3.10
- PyTorch 2.8.0
- MediaPipe 0.10.14
- OpenCV 4.8.0
- NumPy 1.26.4
- CustomTkinter 6.0.0
- eSpeak-NG
- NVIDIA Jetson Orin Nano 8GB

## Model
- Architecture: Multi-Layer Perceptron (MLP)
- Input features: 88
- Hidden layers: 256 and 128 neurons
- Output classes: 28 (A-Z, Delete and Space)

## Run
Activate the project virtual environment and execute:
python signspeak_app.py

## Notes
- Designed and tested on NVIDIA Jetson Orin Nano with JetPack 6.2.1.
- Supports static alphabet gestures, not dynamic sign sequences or digits.
- The training dataset is not included.
