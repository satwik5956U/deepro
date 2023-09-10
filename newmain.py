import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load and preprocess the dataset using ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_dir = 'data/train/'  # Replace with the path to your training directory
test_dir = 'data/test/'    # Replace with the path to your testing directory

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    class_mode='categorical',
    subset='training',
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    class_mode='categorical',
    subset='validation',
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=64,
    class_mode='categorical',
    shuffle=False,
)

# Define the CNN model
model = keras.Sequential(
    [
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", input_shape=(48, 48, 3)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(256, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(7, activation="softmax"),  # 7 emotion classes
    ]
)

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
model.fit(train_generator, validation_data=validation_generator, epochs=20)

# Save the trained model
model.save("emotion_recognition_model.h5")

# Load the trained model for implementation
loaded_model = keras.models.load_model("emotion_recognition_model.h5")

# Implement the model on test data
y_pred = loaded_model.predict(test_generator)
y_pred_classes = y_pred.argmax(axis=-1)
y_true_classes = test_generator.classes

# Calculate accuracy and classification report
accuracy = (y_pred_classes == y_true_classes).mean()
classification_report_result = classification_report(y_true_classes, y_pred_classes)

print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report_result)
