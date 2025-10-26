import cv2
import mediapipe as mp
import numpy as np
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
def extract_feature_vector(results):
    feature_vector = np.zeros(21 * 3 * 2) 
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_vector = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            if len(feature_vector[feature_vector==0]) > len(hand_vector): 
                start_index = np.where(feature_vector == 0)[0][0]
                feature_vector[start_index:start_index+len(hand_vector)] = hand_vector      
    return feature_vector
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)
        vector = extract_feature_vector(results)
        cv2.rectangle(image, (0, 0), (600, 40), (245, 117, 16), -1)
        cv2.putText(image, f'Feature Vector Shape: {vector.shape}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Real-Time Feature Extraction', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()