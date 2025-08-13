# Object-Detection-Recognition-Framework-Visually-Impaired-
Python-based tool that helps visually-impaired people recognize objects and Indian currency notes using deep learning. It uses SSD300 for object detection and Inception V3 for currency classification, all through a simple graphical interface. The system makes daily tasks easier and more accessible for users with visual impairments.

# Deep Learning based Object Detection and Recognition Framework for the Visually-Impaired

## Overview
This project is an assistive tool that uses deep learning to help visually-impaired individuals recognize objects and Indian currency notes. It combines the SSD300 model for object detection and a custom Inception V3 model for currency classification. The application features a simple Tkinter-based interface for easy interaction.

## Features
- Detects and labels multiple objects in images using SSD300
- Recognizes Indian currency notes (fifty, hundred, five hundred, thousand, ten, twenty) using Inception V3
- User-friendly GUI for loading images and viewing results
- Visualizes model training accuracy and loss

## Technologies Used
- Python 3.7
- TensorFlow 1.14.0
- Keras 2.3.1
- OpenCV
- Matplotlib
- Tkinter

## Installation
1. Clone this repository.
2. (Recommended) Create and activate a Python 3.7 virtual environment.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Ensure the `model/` and `Dataset/` folders are present with the required files.

## Usage
1. Run the application:
   ```
   python ObjectDetection.py
   ```
2. Use the GUI to load images, run object detection, and view results.

## Project Structure
- `ObjectDetection.py` - Main application with GUI
- `test1.py` - Model training and testing script
- `model/` - Pre-trained models and weights
- `Dataset/` - Image dataset for training/testing
- `testImages/` - Sample images for detection
- `requirements.txt` - Python dependencies

## Acknowledgements
- SSD300 and Inception V3 architectures
- OpenCV and Keras communities

## License
This project is for educational and research purposes only.

