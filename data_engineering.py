import mediapipe as mp
import cv2 
import csv 
import numpy as np
import os

# Create an empty .csv file with columns class, x1, y1, z1, x2, y2 ... 

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_pose = mp.solutions.pose # Mediapipe Solutions

cap = cv2.imread('Input/angle.png')

with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.85) as pose:
  image = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB) 
  
  # Make Detections
  results = pose.process(image)

  # Recolor image back to BGR for rendering
  image.flags.writeable = True   
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

  # 4. Pose Detections
  mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                            )
  
  num_coords = len(results.pose_landmarks.landmark)
  landmarks = ['class']
  
  for val in range(1, num_coords+1):
      landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]


  with open('coords.csv', mode='w', newline='') as f:
      csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
      csv_writer.writerow(landmarks)


curl_list = [i for i in os.listdir('Input/Custom') if i.startswith('curl')]
overhead_list = [i for i in os.listdir('Input/Custom') if i.startswith('overhead')]
bench_list = [i for i in os.listdir('Input/Custom') if i.startswith('bench')]
stand = [i for i in os.listdir('Input/Custom') if i.startswith('stand')]
walk = [i for i in os.listdir('Input/Custom') if i.startswith('walk')]
stand_walk = stand + walk

# Writing into .csv the data from curls video extracted using MediaPipe 

class_name = "curl"

for i in curl_list:
  mp_drawing = mp.solutions.drawing_utils # Drawing helpers
  mp_holistic = mp.solutions.pose # Mediapipe Solutions

  cap = cv2.VideoCapture('Input/'+ i)
  fps = int(cap.get(cv2.CAP_PROP_FPS))

  print(fps)
  # Initiate holistic model
  with mp_holistic.Pose(static_image_mode=True, min_detection_confidence=0.85) as holistic:
      
      while cap.isOpened():
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
          mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                  )
          # Export coordinates
          try:
              # Extract Pose landmarks
              pose = results.pose_landmarks.landmark
              row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                        
              # Append class name 
              row.insert(0, class_name)
              
              # Export to CSV
              with open('coords.csv', mode='a', newline='') as f:
                  csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                  csv_writer.writerow(row) 
              
          except:
              pass                  

  cap.release()


class_name = "overhead"

for i in overhead_list:
  mp_drawing = mp.solutions.drawing_utils # Drawing helpers
  mp_holistic = mp.solutions.pose # Mediapipe Solutions

  cap = cv2.VideoCapture('Input/'+ i)
  fps = int(cap.get(cv2.CAP_PROP_FPS))

  print(fps)
  # Initiate holistic model
  with mp_holistic.Pose(static_image_mode=True, min_detection_confidence=0.85) as holistic:
      
      while cap.isOpened():
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
          mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                  )
          # Export coordinates
          try:
              # Extract Pose landmarks
              pose = results.pose_landmarks.landmark
              row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                        
              # Append class name 
              row.insert(0, class_name)
              
              # Export to CSV
              with open('coords.csv', mode='a', newline='') as f:
                  csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                  csv_writer.writerow(row) 
              
          except:
              pass                  

  cap.release()


class_name = "stand/walk"

for i in stand_walk:
  mp_drawing = mp.solutions.drawing_utils # Drawing helpers
  mp_holistic = mp.solutions.pose # Mediapipe Solutions

  cap = cv2.VideoCapture('Input/'+ i)
  fps = int(cap.get(cv2.CAP_PROP_FPS))

  print(fps)
  # Initiate holistic model
  with mp_holistic.Pose(static_image_mode=True, min_detection_confidence=0.85) as holistic:
      
      while cap.isOpened():
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
          mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                  )
          # Export coordinates
          try:
              # Extract Pose landmarks
              pose = results.pose_landmarks.landmark
              row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                        
              # Append class name 
              row.insert(0, class_name)
              
              # Export to CSV
              with open('coords.csv', mode='a', newline='') as f:
                  csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                  csv_writer.writerow(row) 
              
          except:
              pass                  

  cap.release()
  

class_name = "bench"

import mediapipe as mp
import cv2 
import csv
import pandas as pd
import numpy as np
import os
from google.colab.patches import cv2_imshow

for i in ['walking6.mp4', 'walking7.mp4', 'walking8.mp4']:
  mp_drawing = mp.solutions.drawing_utils # Drawing helpers
  mp_holistic = mp.solutions.pose # Mediapipe Solutions

  cap = cv2.VideoCapture('Input/Custom/'+i)
  fps = int(cap.get(cv2.CAP_PROP_FPS))

  print(fps)
  # Initiate holistic model
  with mp_holistic.Pose(static_image_mode=True, min_detection_confidence=0.7) as holistic:
      
      while cap.isOpened():
          ret, frame = cap.read()
              
          if not ret:
            break
          
          # Recolor Feed
          image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          image.flags.writeable = False        
          
          # Make Detections
          results = holistic.process(image)
          # print(results.face_landmarks)
          
          # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
          
          # Recolor image back to BGR for rendering
          image.flags.writeable = True   
          image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

          # 4. Pose Detections
          mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                  )
          # Export coordinates
          try:
              # Extract Pose landmarks
              pose = results.pose_landmarks.landmark
              row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
              
              # Append class name 
              row.insert(0, class_name)
              
              # Export to CSV
              with open('BigData.csv', mode='a', newline='') as f:
                  csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                  csv_writer.writerow(row) 
              
          except Exception as e:
              print(e)     

  cap.release()
