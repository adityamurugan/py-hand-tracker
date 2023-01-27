from gettext import find
import tensorflow as tf
import cv2
import mediapipe as mp
from one_euro_filter import OneEuroFilter
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
from websocket import create_connection
import numpy as np
import math
# TO DO -- remove unused imports
from pyquaternion import Quaternion
from math import acos, atan2, cos, pi, sin
from numpy import array, cross, dot, float64, hypot, zeros
from numpy.linalg import norm
from random import gauss, uniform
from eulerangles import matrix2euler
from kalman import OnlineUnivariateKalmanFilter

# Connections to calculate finger angles
connections = [[0,5,6],[5,6,7],[6,7,8],[0,9,10],[9,10,11],[10,11,12],[0,17,18],[17,18,19],[18,19,20],[0,13,14],[13,14,15],[14,15,16],[1,2,3],[2,3,4]]

# Function to calculate angle between 3 3D points
def angle(a, b, c):
    #create vectors
    ba = a - b
    bc = c - b
    #find axis and angle
    axis = np.cross(ba,bc)/(np.linalg.norm(np.cross(ba,bc)))
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return axis,np.degrees(angle)

# Function to round up x by y
def roundup(x,y):
  return float(math.ceil(x / y)) * y

firstPoints = []

# Initialize kalman filters
filters = [OnlineUnivariateKalmanFilter() for i in range(9)]
filterInitPoints= []
for x in range(12):
    filterInitPoints.append([[]])
# filterInitPoints = [[], [], [], [], [], [], [], [], []]
# Get webcam input
cap = cv2.VideoCapture(0)

# Load gesture recognizer model
model = tf.keras.models.load_model('models/keypoint_classifier.hdf5')
modelL = tf.keras.models.load_model('./models/keypoint_classifier_L.hdf5')

# Initialize mediapipe hands
k=0
with mp_hands.Hands(
    model_complexity=1,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    gesOutput = 0
    gesOutputL = 0
    temp= []
    # Connect to local websocket server - Server should be running before running script
    # TO DO -- Add 'try except' blocks to handle errors
    ws = create_connection("ws://localhost:7000/websocket")
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    dists =[]
    rows = []
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_h, img_w, img_c = image.shape
    if results.multi_hand_landmarks:
      for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
        j=0
        row = []
        row.append('Left') if results.multi_handedness[index].classification[0].label=='Right' else row.append('Right')
        for lmark in hand_landmarks.landmark:
            # Collecting datapoints for gesture inference
            if j == 0:
                indX = lmark.x*img_w
                indY = lmark.y*img_h
            row.append((lmark.x*img_w)-indX)
            row.append((lmark.y*img_h)-indY)
            j+=1
        rows.append(row)
        for row in rows:
          if(row[0]=='Right'):
            maxValue = abs(max(row[1:], key=abs))
            newRow = [x / maxValue for x in row[1:]]
            predict_result = model.predict(np.array([newRow]))
            gesOutput = np.argmax(np.squeeze(predict_result))
          if(row[0]=='Left'):
            maxValue = abs(max(row[1:], key=abs))
            newRow = [x / maxValue for x in row[1:]]
            predict_result = modelL.predict(np.array([newRow]))
            gesOutputL = np.argmax(np.squeeze(predict_result))
        # Getting position of KPs 5 and 13 wrt wrist    
        rt_mid_mcp = hand_landmarks.landmark[13]
        rt_ind_mcp = hand_landmarks.landmark[5]
        rt_wr = hand_landmarks.landmark[0]
        zPt1 = hand_landmarks.landmark[5]
        zPt2 = hand_landmarks.landmark[17]
        zDist = (np.linalg.norm(np.array([zPt1.x,zPt1.y,zPt1.z])-np.array([zPt2.x,zPt2.y,zPt2.z])))*10
        # print(roundup(zDist,0.1))
        dist = [rt_mid_mcp.x-rt_wr.x,rt_mid_mcp.y-rt_wr.y,rt_mid_mcp.z-rt_wr.z,
                rt_ind_mcp.x-rt_wr.x,rt_ind_mcp.y-rt_wr.y,rt_ind_mcp.z-rt_wr.z]
        dist.extend([-round(hand_landmarks.landmark[0].x-0.5,1),-round(hand_landmarks.landmark[0].y-0.5,2),round(zDist,1)-1.5])
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
      # for hand_landmarks in results.multi_hand_world_landmarks:
        # Getting angles of fingers
        i = 0
        for connection in connections:
          pt1 = np.array([hand_landmarks.landmark[connection[0]].x,hand_landmarks.landmark[connection[0]].y,hand_landmarks.landmark[connection[0]].z])
          base = np.array([hand_landmarks.landmark[connection[1]].x,hand_landmarks.landmark[connection[1]].y,hand_landmarks.landmark[connection[1]].z])
          pt2 = np.array([hand_landmarks.landmark[connection[2]].x,hand_landmarks.landmark[connection[2]].y,hand_landmarks.landmark[connection[2]].z])
          ax,fingerAngle = angle(pt1,base,pt2)
          if(connection[0] == 0 and connection[1]!=17):
            fingerAngle = fingerAngle+70
          elif(connection[0] == 0 and connection[1]==17):
            fingerAngle = fingerAngle + 10
          else:
            fingerAngle = fingerAngle - 180
          dist.append(fingerAngle)
          if(i == 11):
            dist.append(55.184)
          i = i + 1
        if k==0:
          x_track = [OneEuroFilter(k, dist[p], 1.6, min_cutoff=0.0001,beta=1.0,d_cutoff=0.5) for p in range(12)]
        if k > 1:
          for i in range(12):
            dist[i] = round(x_track[i](k, dist[i]),2)
        if(k<200):
          for a,b in enumerate(filterInitPoints):
            b.append(dist[a])
        k+=1
        # if(k>200):
        #   k=0
        print(results.multi_handedness[index].classification[0].label,results.multi_handedness[index].classification[0].score)
        if results.multi_handedness[index].classification[0].label=='Right' and results.multi_handedness[index].classification[0].score>0.85:
          dist.extend([gesOutputL,1])
        else:
          dist.extend([gesOutput,0])
        print(dist[len(dist)-1])
        dists.append(dist)
    if len(dists)!=0:
      # print(dists)
      ws.send(str(dists))
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    ws.close()
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
