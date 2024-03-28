import os
import winsound

import cv2
import numpy as np

from .LoadModel import AccidentDetectionModel

json=r'.\detection_app\model.json'
model_weight=r'.\detection_app\model_weights.h5'
model = AccidentDetectionModel(json, model_weight)
font = cv2.FONT_HERSHEY_SIMPLEX
frequency = 2000
duration = 1000
output_directory = "accident_frames"
def startapplication():
    video = cv2.VideoCapture(r".\video1.mp4") # for camera use video = cv2.VideoCapture(0)
    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            print("frame reached end")
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(gray_frame, (250, 250))

        pred, prob = model.predict_accident(roi[np.newaxis, :, :]) 
        if(pred == "Accident"):
            prob = (round(prob[0][0]*100, 2))
            
            # to beep when alert:
            if(98.5<= prob <= 99.6):
                # winsound.Beep(frequency, duration)
                print('accident detected')
                # Save the frame as an image
                frame_path = os.path.join(output_directory, f"accident_frame_{frame_count}.jpg")
                cv2.imwrite(frame_path, frame)
                print(f"Accident frame saved: {frame_path}")
                frame_count += 1
                return 1

            cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, pred+" "+str(prob), (20, 30), font, 1, (255, 255, 0), 2)

        # if cv2.waitKey(33) & 0xFF == ord('q'):
        #     return
        # cv2.imshow('Video', frame)  

