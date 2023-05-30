import streamlit as st
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import time
import pygame
from pygame import mixer as pyg_mixer

# Load pre-trained model
model_best = torch.jit.load('model_best1510.pt', map_location=torch.device('cpu'))
classes = ['Drinking', 'Others', 'Smoking', 'Talking on Phone']
transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define function to preprocess image for model input and make predictions
def predict(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert to RGB format
    image = Image.fromarray(image) # convert to PIL Image object
    image = transformer(image).unsqueeze(0) # preprocess image
    with torch.no_grad():
        output = torch.nn.functional.softmax(model_best(image)[0], dim=0) # make predictions
        confidence, prediction = torch.max(output, 0) # get class with highest confidence
    return classes[prediction], float(confidence)

# Set up camera capture
cap = cv2.VideoCapture(0)

# Define Streamlit app
st.title('Distracted Driver Detection')
st.write('Click the button below to start capturing images from your camera')
pygame.init()
# pygame.mixer.init() # initialize pygame mixer
# mixer.init()
pyg_mixer.init()

alarm_sound = pygame.pyg_mixer.Sound('198841__bone666138__analog-alarm-clock.wav') # load alarm sound

if st.button('Capture'):
    # Create a frame for displaying the video stream
    video_frame = st.empty()

    start_time = time.time() # set start time
    alarm_triggered = False # set alarm triggered flag

    while True:
        ret, frame = cap.read() # capture image from camera
        if ret:
            predicted_class, confidence = predict(frame) # make predictions on captured image
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert back to RGB format
            frame = cv2.resize(frame, (500, 500)) # resize for display
            # Display the video stream and predicted class with highest confidence in the frame
            video_frame.image(frame, caption=f'Predicted class: {predicted_class}, Confidence: {confidence:.2f}') 
            
            # Check if activity is detected
            if predicted_class in ['Drinking', 'Smoking', 'Talking on Phone'] and not alarm_triggered:
                if time.time() - start_time >= 2: # wait for 5 seconds
                    alarm_sound.play() # play alarm sound
                    alarm_triggered = True # set alarm triggered flag
            else:
                start_time = time.time() # reset start time
                alarm_triggered = False # reset alarm triggered flag

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release() # release camera
cv2.destroyAllWindows() # close all windows
