from keras.preprocessing.image import ImageDataGenerator

# Create an ImageDataGenerator for the training dataset; xs;
train_datagen = ImageDataGenerator(rescale=1./255)


# Create an ImageDataGenerator for the validation dataset
val_datagen = ImageDataGenerator(rescale=1./255)

# Option 1: Using raw strings
train_generator = train_datagen.flow_from_directory(
    r'C:\Users\Ali\OneDrive - University of Greenwich\Pictures\archive (6)\train'll
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)
#raw strings method

# Option 2:Escaping the backslashess
val_generator = val_datagen.flow_from_directory(
    'C:\\Users\\Ali\\OneDrive - University of Greenwich\\Pictures\\archive (6)\\val',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))  # Use 2 here because you have 2 classes

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Train the model on the training data
model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=10,
    validation_data=val_generator,
    validation_steps=50
)# defining model parameters here

# Evaluate the model on the validation data,,
val_loss, val_acc = model.evaluate(val_generator)
print('Validation accuracy:', val_acc)

import numpy as np
import random

# Get a random batch of images from the validation generator
random_batch = random.choice(val_generator)

# Get the images and true labels from the random batch
images, true_labels = random_batch

# Make predictions on the images
predictions = model.predict(images)

# Convert the predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Get the class labels from the generator
class_labels = list(val_generator.class_indices.keys())

# Convert the true labels and predicted labels to their corresponding class names
true_class_names = [class_labels[int(label)] for label in true_labels]
predicted_class_names = [class_labels[int(label)] for label in predicted_labels]

# Print the true and predicted class names
for true, predicted in zip(true_class_names, predicted_class_names):
    print(f"True: {true}, Predicted: {predicted}")

import matplotlib.pyplot as plt

# Get a random batch of images from the validation generator
random_batch = random.choice(val_generator)

# Get the images and true labels from the random batch
images, true_labels = random_batch

# Make predictions on the images
predictions = model.predict(images)

# Convert the predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Get the class labels from  generator
class_labels = list(val_generator.class_indices.keys())

# Convert the true labels and predicted labels to their corresponding class names
true_class_names = [class_labels[int(label)] for label in true_labels]
predicted_class_names = [class_labels[int(label)] for label in predicted_labels]

# Display the images along with their predicted and actual labels
for i in range(len(images)):
    plt.imshow(images[i])
    plt.title(f"True: {true_class_names[i]}, Predicted: {predicted_class_names[i]}")
    plt.show()    

# Train the model on the training data and capture the history.
history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=10,
    validation_data=val_generator,
    validation_steps=50
)

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
