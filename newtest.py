import cv2
import numpy as np
from tensorflow import keras

# Load the trained emotion recognition model
emotion_model = keras.models.load_model("emotion_recognition_model.h5")

# Define the emotions list corresponding to model output classes
emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Function to detect and recognize emotions from frames
def detect_emotion(frame):
    # Convert grayscale frame to RGB (replicate the single channel three times)
    #faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

    # Preprocess the frame
    resized_frame = cv2.resize(frame_rgb, (48, 48))
    input_frame = np.expand_dims(resized_frame, axis=0)
    input_frame = input_frame.astype("float32") / 255.0

    # Predict the emotion
    predictions = emotion_model.predict(input_frame)
    predicted_class = np.argmax(predictions[0])
    predicted_emotion = emotions[predicted_class]

    return predicted_emotion

# Capture video from the camera
camera = cv2.VideoCapture(0)
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #faces= faceDetect.detectMultiScale(gray_frame, 1.3, 3)

    # Detect and recognize emotions
    predicted_emotion = detect_emotion(gray_frame)

    # Display the emotion on the frame
    cv2.putText(frame, predicted_emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Emotion Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
