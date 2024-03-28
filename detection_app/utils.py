import os
from django.core.files.storage import FileSystemStorage
from accident_detection.settings import VIDEO_UPLOAD_PATH
import shutil

import cv2
import face_recognition
import pickle


face_path = "model/path"


def handle_uploaded_file(file):
    shutil.rmtree(VIDEO_UPLOAD_PATH)
    if not os.path.exists(VIDEO_UPLOAD_PATH):
        os.makedirs(VIDEO_UPLOAD_PATH)
    fs = FileSystemStorage(VIDEO_UPLOAD_PATH) #defaults to   MEDIA_ROOT  
    filename = fs.save(file.name, file)
    # print('saved')
    file_url = fs.url(filename)
    video_path = VIDEO_UPLOAD_PATH +'/'+ file.name

    # prediction = detectFakeVideo(video_path)


def mark_attendance(timePeriod):
    webcam = cv2.VideoCapture(0) 
    while True:
    # Loop until the camera is working
        rval = False
        while (not rval):
            # Put the image from the webcam into 'frame'
            (rval, frame) = webcam.read()
            if (not rval):
                print("Failed to open webcam. Trying again...")

        # Flip the image (optional)
        frame = cv2.flip(frame, 1)  # 0 = horizontal ,1 = vertical , -1 = both
        frame_copy = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        predictions = predict(frame_copy, model_path="")  # add path here
        font = cv2.FONT_HERSHEY_DUPLEX
        for name, (top, right, bottom, left) in predictions:
            top *= 4  # scale back the frame since it was scaled to 1/4 in size
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
            cv2.putText(frame, name, (left - 10, top - 6), font, 0.8, (255, 255, 255), 1)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()
    return name


def predict(img_path, knn_clf=None, model_path=None, threshold=0.5):  # 6 needs 40+ accuracy, 4 needs 60+ accuracy
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")
    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(face_path, 'rb') as f:
            knn_clf = pickle.load(f)
    # Load image file and find face locations
    img = img_path
    face_box = face_recognition.face_locations(img)
    # If no faces are found in the image, return an empty result.
    if len(face_box) == 0:
        return []
    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(img, known_face_locations=face_box)
    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=2)
    matches = [closest_distances[0][i][0] <= threshold for i in range(len(face_box))]
    
    # print(closest_distances)
    
    
    
    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
            zip(knn_clf.predict(faces_encodings), face_box, matches
                )]
import winsound

import cv2
import numpy as np

from .LoadModel import AccidentDetectionModel

json=r'.\detection_app\accident_detection_model\model.json'
model_weight=r'.\detection_app\accident_detection_model\model_weights.h5'
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

