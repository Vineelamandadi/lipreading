import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Constants
NUM_FRAMES = 21
HEIGHT, WIDTH = 64, 64
CHANNELS = 3

# Provided labels
labels_list = ['a', 'bye', 'can', 'cat', 'demo', 'dog', 'hello', 'here', 'is', 'lips', 'my', 'read', 'you']

# Create a word to index mapping globally
word_to_index = {word: i for i, word in enumerate(labels_list)}
index_to_word = {i: word for word, i in word_to_index.items()}

# Function to decode predictions
def decode_predictions(predictions):
    decoded = np.argmax(predictions, axis=-1)
    sentence = []
    for frame_predictions in decoded:
        word = index_to_word[frame_predictions[0]]
        if len(sentence) == 0 or sentence[-1] != word:
            sentence.append(word)
    return ' '.join(sentence)

# Load the trained model
model = load_model('lip_reading3.h5')

# Real-time video processing using the trained model
video_path = r"C:\MINE\miniproject-b05\inputs\kruthi.mp4"
cap = cv2.VideoCapture(video_path)
frames = []
sentence = ""
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        print("No face detected. Stopping...")
        break

    resized_frame = cv2.resize(frame, (WIDTH, HEIGHT))
    frames.append(resized_frame)
    if len(frames) == NUM_FRAMES:
        input_frames = np.expand_dims(frames, axis=0)
        input_frames = input_frames / 255.0  # Normalize the frames
        predicted_sentence = model.predict(input_frames)
        sentence = decode_predictions(predicted_sentence)
        print('Recognized Sentence:', sentence)
        frames = []  # Reset frames list

    # Display the frame with the predicted sentence
    cv2.putText(frame, sentence, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Lip Reading', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
