import os
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, LSTM, TimeDistributed, Bidirectional, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
NUM_FRAMES = 21  # Since there are 21 screenshots per folder
HEIGHT, WIDTH = 64, 64
CHANNELS = 3
NUM_CLASSES = 14  # 14 unique labels

# Provided labels
labels_list = ['a', 'bye', 'can', 'cat', 'demo', 'dog', 'hello', 'here', 'is', 'lips', 'my', 'read', 'you']

# Create a word to index mapping globally
word_to_index = {word: i for i, word in enumerate(labels_list)}

# Function to preprocess screenshots
def preprocess_screenshots(screenshot_files, folder_path):
    frames = []
    for file in sorted(screenshot_files)[:NUM_FRAMES]:
        image_path = os.path.join(folder_path, file)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Could not read image: {image_path}")
            continue
        resized_frame = cv2.resize(frame, (WIDTH, HEIGHT))
        frames.append(resized_frame)

    if len(frames) < NUM_FRAMES:
        print(f"Folder {folder_path} did not have enough frames. Found {len(frames)} frames. Padding with black frames.")
        while len(frames) < NUM_FRAMES:
            frames.append(np.zeros((HEIGHT, WIDTH, CHANNELS), dtype=np.uint8))

    return np.array(frames)

# Load dataset
def load_dataset(dataset_path):
    X = []
    Y = []
    labels = sorted(os.listdir(dataset_path))
    #print(f"Found labels: {labels}")
    for label in labels:
        label_dir = os.path.join(dataset_path, label)
        if os.path.isdir(label_dir):
            word_label = label.split('_')[0]
            screenshot_files = [f for f in os.listdir(label_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            frames = preprocess_screenshots(screenshot_files, label_dir)
            if frames is not None and word_label in word_to_index:
                X.append(frames)
                Y.append(word_label)
            else:
                print(f"Failed to process screenshots in folder: {label_dir}")
        else:
            print(f"'{label_dir}' is not a directory")
    return np.array(X), np.array(Y)

# Path to the dataset
dataset_path = 'C:/MINE/miniproject/dataset/outputs'
X, Y = load_dataset(dataset_path)

# Convert labels to indices
Y = np.array([word_to_index[word] for word in Y])

# Repeat labels for each frame
Y = np.repeat(Y[:, np.newaxis], NUM_FRAMES, axis=1)

# Preprocess labels (convert to categorical)
Y = keras.utils.to_categorical(Y, num_classes=NUM_CLASSES)

# Check if dataset is loaded correctly
print(f"Loaded {X.shape[0]} samples")

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Custom data generator
def custom_data_generator(X, Y, batch_size):
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )
    while True:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            batch_indices = indices[start:end]
            batch_X = X[batch_indices]
            batch_Y = Y[batch_indices]
            augmented_batch_X = np.zeros_like(batch_X)
            for i in range(batch_X.shape[0]):
                for j in range(batch_X.shape[1]):
                    augmented_batch_X[i, j] = datagen.random_transform(batch_X[i, j])
            yield augmented_batch_X, batch_Y

# Split the dataset into training and validation sets
if X.shape[0] > 0:
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Build the enhanced lip-reading model
    def build_model(input_shape, num_classes):
        model = Sequential([
            Input(shape=input_shape),
            TimeDistributed(Conv2D(64, (3, 3), activation='relu')),
            TimeDistributed(BatchNormalization()),
            TimeDistributed(MaxPooling2D(pool_size=(2, 2))),
            TimeDistributed(Conv2D(128, (3, 3), activation='relu')),
            TimeDistributed(BatchNormalization()),
            TimeDistributed(MaxPooling2D(pool_size=(2, 2))),
            TimeDistributed(Conv2D(256, (3, 3), activation='relu')),
            TimeDistributed(BatchNormalization()),
            TimeDistributed(MaxPooling2D(pool_size=(2, 2))),
            TimeDistributed(Flatten()),
            Bidirectional(LSTM(512, return_sequences=True)),
            Dropout(0.5),
            TimeDistributed(Dense(256, activation='relu')),
            TimeDistributed(Dense(num_classes, activation='softmax'))
        ])
        model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    # Build and train the model
    input_shape = (NUM_FRAMES, HEIGHT, WIDTH, CHANNELS)
    model = build_model(input_shape, NUM_CLASSES)

    # Fit the model with custom data generator
    batch_size = 32
    history = model.fit(custom_data_generator(X_train, Y_train, batch_size),
                        epochs=2, 
                        steps_per_epoch=len(X_train) // batch_size,
                        validation_data=(X_val, Y_val))

    # Save the trained model
    model.save('lip_reading3.h5')

    # Plot accuracy graph
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Evaluate the model
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred[:, -1, :], axis=1)
    y_true = np.argmax(Y_val[:, -1, :], axis=1)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels_list, yticklabels=labels_list)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification report
    class_report = classification_report(y_true, y_pred_classes, target_names=labels_list)
    print('Classification Report:')
    print(class_report)

else:
    print("No samples were loaded. Please check the dataset path and structure.")
