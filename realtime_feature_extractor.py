import cv2
import mediapipe as mp
import numpy as np
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
def extract_feature_vector(results):
    lh = np.zeros(21*3)
    rh = np.zeros(21*3)
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_index, hand_info in enumerate(results.multi_handedness):
            hand_label = hand_info.classification[0].label
            hand_landmarks = results.multi_hand_landmarks[hand_index]
            if hand_label == "Left":
                lh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            elif hand_label == "Right":
                rh = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()     
    return np.concatenate([lh, rh])
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
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_index, hand_info in enumerate(results.multi_handedness):
                hand_label = hand_info.classification[0].label
                hand_landmarks = results.multi_hand_landmarks[hand_index]
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)
                wrist = hand_landmarks.landmark[0]
                h, w, _ = image.shape
                x, y = int(wrist.x * w), int(wrist.y * h)
                cv2.putText(image, f'{hand_label} Hand', (x - 50, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        vector = extract_feature_vector(results)
        cv2.rectangle(image, (0, 0), (600, 40), (245, 117, 16), -1)
        cv2.putText(image, f'Feature Vector Shape: {vector.shape}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Real-Time Feature Extraction', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
import json 
def get_all_sign_words(json_path):
    sign_words = []
    seen = set()
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        for entry in data:
            word = entry['gloss']
            if word not in seen:
                sign_words.append(word)
                seen.add(word)
    except FileNotFoundError:
        print(f"Error: {json_path} file not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON file at {json_path}.")
        return []
    return sign_words
if __name__ == '__main__':
    WLASL_PATH = "C:\\Gautam\\Projects\\sign_language_translator\\model_development\\WLASL_v0.3.json"
    all_words = get_all_sign_words(WLASL_PATH)
    if all_words:
        print(f"Total unique words found: {len(all_words)}")
        print("First 20 words:", all_words[:20])
    