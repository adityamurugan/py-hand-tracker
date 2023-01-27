import tensorflow as tf
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
import numpy as np

# Demo file for testing inference of gesture

cap = cv2.VideoCapture(0)
font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 1
fontColor              = (0,255,0)
thickness              = 1
lineType               = 2

model = tf.keras.models.load_model('./models/keypoint_classifier_L.hdf5')

with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.9,
    max_num_hands=2,
    min_tracking_confidence=0.9) as hands:
  while cap.isOpened():
    gesOutput = 0
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    rows = []
    image_height, image_width, _ = image.shape
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
        row=[]
        i=0
        print(results.multi_handedness[index].classification[0].label)
        row.append('Left') if results.multi_handedness[index].classification[0].label=='Right' else row.append('Right')
        for lmark in hand_landmarks.landmark:
            if i == 0:
                indX = lmark.x*image_width
                indY = lmark.y*image_height
                # print(lmark.z)
            row.append((lmark.x*image_width)-indX)
            row.append((lmark.y*image_height)-indY)
            i+=1
        rows.append(row)
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())            
    for row in rows:
      if(row[0]=='Left'):
        maxValue = abs(max(row[1:], key=abs))
        newRow = [x / maxValue for x in row[1:]]
        predict_result = model.predict(np.array([newRow]))
        gesOutput = np.argmax(np.squeeze(predict_result))
    image = cv2.flip(image, 1)
    cv2.putText(image,str(gesOutput), 
    (0,image_height-20), 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()