import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt
import numpy as np

#Preparation of training data.
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0


#Perform image augmentation.
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.9, 1.1],
    fill_mode='nearest'
)

train_generator = train_datagen.flow(train_images[..., np.newaxis], train_labels, batch_size=32)

#Creation of CNN model.
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

#Model training.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_generator, epochs=10, steps_per_epoch=len(train_images) // 32, validation_data=(test_images[..., np.newaxis], test_labels))


#Evaluate model performance.
test_loss, test_accuracy = model.evaluate(test_images[..., np.newaxis], test_labels)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

#Visualize network performance results.
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

#Visualize dataset images.
def show_images(images, labels):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(labels[i])
    plt.show()

show_images(train_images, train_labels)

#Display prediction results of the best-performing model.
def predict(model, test_images, test_labels):
    predictions = model.predict(test_images[..., np.newaxis])
    predicted_labels = np.argmax(predictions, axis=1)

    # Print prediction results and display images
    for i in range(25):
        print(f'Predicted: {predicted_labels[i]}, Actual: {test_labels[i]}')
        plt.figure()
        plt.imshow(test_images[i], cmap=plt.cm.binary)
        plt.xlabel(f'Predicted: {predicted_labels[i]}, Actual: {test_labels[i]}')
        plt.show()

predict(model, test_images[:25], test_labels[:25])
