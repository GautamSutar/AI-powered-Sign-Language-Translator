import json
import os
import cv2
import numpy as np
import requests
from pytube import YouTube
import mediapipe as mp

DATA_PATH = os.path.join('MP_DATA')
WLASL_JSON_PATH = "C:\\Gautam\\Projects\\sign_language_translator\\model_development\\WLASL_v0.3.json"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_features_vector(results):
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

def process_video(video_path, start_frame, end_frame):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None
    sequence_data = []
    if end_frame == -1:
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        for frame_num in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            keypoints = extract_features_vector(results)
            sequence_data.append(keypoints)
        cap.release()
        if not sequence_data:
            return None
        normalized_sequence = []
        indices = np.linspace(0, len(sequence_data) - 1, 30, dtype = int)
        for i in indices:
            normalized_sequence.append(sequence_data[i])

        return np.array(normalized_sequence)
    
def download_and_process_instance(instance):
    gloss = instance['gloss']
    url = instance['url']
    video_id = instance['video_id']
    split = instance['split']
    save_dir = os.path.join(DATA_PATH, split, gloss)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{video_id}.npy')
    if os.path.exists(save_path):
        print(f"Skipping {video_id} for '{gloss}', already processed.")
        return 
    temp_video_path = f"temp_{video_id}.mp4"
    try:
        if 'youtube.com' in url or 'youtu.be' in url:
            yt = YouTube(url)
            stream = yt.streams.filter(file_extensions="mp4", progressive=True).first()
            stream.download(filename = temp_video_path)
        elif '.mp4' in url:
            print(f"Downloading direct MP4: {url}")
            response = requests.get(url, timeout = 15)
            response.raise_for_status()
            with open(temp_video_path, 'wb') as f:
                f.write(response.content)
        else:
            print(f"Skipping unsupported URL type: {url}")
            return
        processed_data = process_video(temp_video_path, instance['frame_start'], instance['frame_end'])

        if processed_data is not None and processed_data.shape == (30, 126):
            np.save(save_path, processed_data)
            print(f"Successfully processed and saved data for '{gloss}' to {save_path}")
        else:
            print(f"Failed to process video {video_id} for '{gloss}'. Data might be empty or malformed.")

    except Exception as e:
        print(f"!!!!!!! FAILED for video {video_id}. Reason: {e} !!!!!!!!")
    
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path) 
            
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
def main():
    all_words = get_all_sign_words(WLASL_JSON_PATH)
    TARGET_WORDS = all_words[:5]
    print("Target words: ", TARGET_WORDS)   
    with open(WLASL_JSON_PATH, 'r') as f:
        content = json.load(f)
    for entry in content:
        if entry['gloss'] in TARGET_WORDS:
            for instance in entry['instances']:
                instance['gloss'] = entry['gloss']
                download_and_process_instance(instance)
if __name__ == "__main__":
    main()