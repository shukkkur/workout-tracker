import pickle 
import pandas as pd
import mediapipe as mp
import cv2
import os 
import sys
import numpy as np
import argparse
from utils import calculate_angle
import imutils

with open('BigData.pkl', 'rb') as f:
    model = pickle.load(f)

# parse parameters
parser = argparse.ArgumentParser()

parser.add_argument("--input_video_path", type=str)
parser.add_argument("--output_video_path", type=str, default="")
parser.add_argument("--draw_pose", type=int, default=1)
parser.add_argument("--info", type=int, default=1)

args = parser.parse_args()

input_video_path = args.input_video_path
output_video_path = args.output_video_path
draw_pose = args.draw_pose
info = args.info

if output_video_path == "":
    # output video in same path
    output_video_path = input_video_path.split('.')[0] + "video_output.mp4"

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.pose # Mediapipe Solutions

cap = cv2.VideoCapture(input_video_path)
output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

print("FPS: ", fps)

# Curl counter variables
i = 0
j = 0 
k = 0
currentFrame = 0
counter1 = 0 
counter2 = 0 
counter3 = 0

if imutils.is_cv2() is True :
  prop = cv2.cv.CV_CAP_PROP_FRAME_C
else:
  prop = cv2.CAP_PROP_FRAME_COUNT
total = int(cap.get(prop))

stage1 = None
stage2 = None
stage3 = None

# Initiate holistic model
with mp_holistic.Pose(static_image_mode=True, min_detection_confidence=0.75) as holistic:
  while cap.isOpened():
    print('Processed: {}%'.format(round( (currentFrame / total) * 100, 2)))
    ret, frame = cap.read()

    if not ret:
      break
    
    # Recolor Feed
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False        
    
    # Make Detections
    results = holistic.process(image)
    
    # Recolor image back to BGR for rendering
    image.flags.writeable = True   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Pose Detections
    if draw_pose:
      mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                )
    # Export coordinates
    try:
      # Extract Pose landmarks
      landmarks = results.pose_landmarks.landmark
      row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in landmarks]).flatten())

      # Make Detections
      X = pd.DataFrame([row])
      body_language_class = model.predict(X)[0]
      body_language_prob = model.predict_proba(X)
      # print('Predicted ', body_language_class)

      # Get coordinates
      if body_language_class == 'overhead':
        if landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].visibility > landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].visibility:
          hand = 'left '
          shoulder = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
          elbow = [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]
          wrist = [landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y]
          hip = [landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].y]
          elbow = [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]
        else:
          hand = 'right '
          shoulder = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y]
          elbow = [landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].y]
          wrist = [landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y]
          hip = [landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].y]
          elbow = [landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].y]
      

      if body_language_class == 'curl':
        if landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].visibility > landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].visibility:
          hand = 'left '
          shoulder = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
          elbow = [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]
          wrist = [landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y]
          hip = [landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].y]
          elbow = [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]
        else:
          hand = 'right '
          shoulder = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y]
          elbow = [landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].y]
          wrist = [landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y]
          hip = [landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].y]
          elbow = [landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].y]
      

      if body_language_class == 'bench':
        if landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].visibility > landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].visibility:
          hand = 'left '
          shoulder = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]
          elbow = [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]
          wrist = [landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y]
          hip = [landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].y]
          elbow = [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]
        else:
          hand = 'right '
          shoulder = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y]
          elbow = [landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].y]
          wrist = [landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y]
          hip = [landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].y]
          elbow = [landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].y]


      # Calculate the angle 
      angle1 = calculate_angle(shoulder, elbow, wrist) # elbow
      if angle1 > 110 :
        i += 1
        stage1 = "down"
      if (angle1 < 90 and stage1 =='down') and body_language_class == 'curl' and i >= 10: #and body_language_prob[0][0] > 0.95:
        stage1 ="up"
        i = 0 
        counter1 += 1
      if info:
        cv2.putText(image, 
              str(angle1), 
              tuple(np.multiply(elbow, [output_width, output_height]).astype(int)), 
              cv2.FONT_HERSHEY_SIMPLEX, 
              0.5, 
              (255, 255, 255), 
              2, 
              cv2.LINE_AA) # elbow

      #####################################################################

      angle2 = calculate_angle(elbow, shoulder, hip)   # shoulder
      if angle2 < 100:
        k += 1
        stage2 = "down"
      if (angle2 > 120 and stage2 =='down') and body_language_class == 'overhead' and k >= 10:
        k = 0
        stage2 ="up"
        counter2 += 1 
      
      if info:
        cv2.putText(image, 
              str(angle2), 
              tuple(np.multiply(shoulder, [output_width, output_height]).astype(int)), 
              cv2.FONT_HERSHEY_SIMPLEX, 
              0.5, 
              (255, 255, 255), 
              2, 
              cv2.LINE_AA) # shoulder

      # brench press
      if angle1 > 120:
        j += 1
        stage3 = "up"
      if angle1 < 90 and (stage3 =='up' and body_language_class == 'bench') and j >= 10: #and body_language_prob[0][0] > 0.95:
        stage3 ="down"
        j = 0 
        counter3 += 1
        

      # Probabilities 
      if info:
        cv2.rectangle(image,(output_width//2-100, 100),((output_width//2)+150, 160), (245, 117, 16), -1)
        cv2.putText(image, str(body_language_prob) + ' ' + hand, (output_width//2-80,130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA)        
        
    except Exception as e:
      exc_type, exc_obj, exc_tb = sys.exc_info()
      print(e, end=' ') 
      print('Line number: ' + str(exc_tb.tb_lineno))
    
    # Curl 
    cv2.rectangle(image, (0,0), (100, 80), (245, 117, 16), -1)
    cv2.putText(image, 'curls', (15,15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2, cv2.LINE_AA) # LEFT WORD
    cv2.putText(image, str(counter1), (5,75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA) # LEFT NUMBER

    # Overhead
    cv2.rectangle(image, (output_width-100,0), (output_width, 80), (245, 117, 16), -1) 
    cv2.putText(image, 'overhead', (output_width-100,15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA) # LEFT WORD
    cv2.putText(image, str(counter2), (output_width-95,75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA) # LEFT NUMBER

    # Bench
    cv2.rectangle(image, (output_width//2-50, 0), ((output_width//2)+50, 80), (245, 117, 16), -1) 
    cv2.putText(image, 'bench', (output_width//2-45, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2, cv2.LINE_AA) # MIDDLe WORD
    cv2.putText(image, str(counter3), (output_width//2-45, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA) # MIDDLe NUMBER
    output_video.write(image)
    currentFrame += 1

cap.release()
output_video.release()