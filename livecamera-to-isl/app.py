from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import os
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Directory to save gesture data
if not os.path.exists('gesture_data'):
    os.makedirs('gesture_data')

# Mediapipe Hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=4, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Route for the main webpage
@app.route('/')
def index():
    return render_template('index.html')

# API for capturing gestures
@app.route('/capture', methods=['POST'])
def capture_gesture():
    label = request.json['label']
    if label:
        capture_hand_data(label)
        return jsonify({'message': f'Gesture data for "{label}" has been captured!'})
    return jsonify({'message': 'Error capturing gesture.'})

def capture_hand_data(label):
    cap = cv2.VideoCapture(0)
    data = []
    
    while True:
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])
                data.append(np.array(landmarks).flatten())

        # Show the camera frame
        cv2.imshow("Capture Hand Data - Press 'q' to stop", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # Save data as .npy file
    data = np.array(data)
    np.save(f'gesture_data/{label}.npy', data)

# API for training the model
@app.route('/train', methods=['POST'])
def train_model():
    data = []
    labels = []
    gesture_data_dir = 'gesture_data'

    for file in os.listdir(gesture_data_dir):
        if file.endswith('.npy'):
            gesture = np.load(os.path.join(gesture_data_dir, file))
            label = file.split('.')[0]
            data.append(gesture)
            labels.extend([label] * len(gesture))

    data = np.concatenate(data, axis=0)
    labels = np.array(labels)

    unique_labels = np.unique(labels)
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    y = np.array([label_to_int[label] for label in labels])

    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)

    model = models.Sequential([
        layers.Input(shape=(63,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(unique_labels), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
    model.save('sign_language_custom_model.h5')

    return jsonify({'message': 'Model has been trained and saved!'})

# API for recognizing gestures
@app.route('/recognize', methods=['POST'])
def recognize_gesture():
    cap = cv2.VideoCapture(0)

    try:
        model = load_model('sign_language_custom_model.h5')
    except:
        return jsonify({'message': 'Please train the model first!'})

    gesture_data_dir = 'gesture_data'
    unique_labels = [file.split('.')[0] for file in os.listdir(gesture_data_dir) if file.endswith('.npy')]
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    int_to_label = {idx: label for label, idx in label_to_int.items()}

    sentence = []

    def predict_gesture(landmarks):
        landmarks = np.array(landmarks).flatten().reshape(1, -1)
        predictions = model.predict(landmarks)
        predicted_label = int_to_label[np.argmax(predictions)]
        return predicted_label

    while True:
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])

                gesture = predict_gesture(landmarks)

                if gesture not in sentence:
                    sentence.append(gesture)

                if len(sentence) > 7:
                    sentence.pop(0)

                sentence_display = ' '.join(sentence)
                cv2.putText(frame, f'Sentence: {sentence_display}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Real-Time Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return jsonify({'message': 'Gesture recognition complete'})

if __name__ == '__main__':
    app.run(debug=True)
