import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Step 1: Set parameters
img_height, img_width = 64, 64
batch_size = 32
data_dir = 'asl_alphabet_train'  # ✅ Make sure this folder exists and contains all 29 classes

# Step 2: Preprocess the images
# ✅ Rescale and split into training and validation from the same folder
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# ✅ Training data generator
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # ✅ Ensures one-hot encoded labels
    subset='training'
)

# ✅ Validation data generator
val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # ✅ Must match output activation
    subset='validation'
)

# ✅ Step 3: Build the CNN model
model = Sequential([
    tf.keras.Input(shape=(img_height, img_width, 3)),  # ✅ Best practice input
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(29, activation='softmax')  # 🛠️ FIXED: 29 classes instead of 26
])

# Step 4: Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # ✅ Must match class_mode='categorical'
    metrics=['accuracy']
)

# Step 5: Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# Step 6: Evaluate the model
loss, accuracy = model.evaluate(val_generator)
print(f"✅ Validation Accuracy: {accuracy * 100:.2f}%")

# Step 7: Save the trained model
model.save("asl_alphabet_model.h5")
print("✅ Model saved as asl_alphabet_model.h5")
