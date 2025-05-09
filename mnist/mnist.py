from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pygame
import numpy as np
import cv2
from scipy import ndimage

model_path = "mnist/mnist_model.h5"

def get_model():
    if os.path.exists(model_path):
        return load_model(model_path)
    
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train_categorical = to_categorical(y_train, 10)
    y_test_categorical = to_categorical(y_test, 10)
    
    # CNN model
    datagen = ImageDataGenerator(
        rotation_range = 10,
        zoom_range = 0.1,
        width_shift_range = 0.1,
        height_shift_range = 0.1
    )

    # Build model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(datagen.flow(x_train, y_train_categorical), epochs=20, validation_data=(x_test, y_test_categorical))

    # Save model
    model.save(model_path)
    return model

def center_digits(image):
    # Crop non-zero borders
    while np.sum(image[0]) == 0: image = image[1:]
    while np.sum(image[-1]) == 0: image = image[:-1]
    while np.sum(image[:, 0]) == 0: image = image[:, 1:]
    while np.sum(image[:, -1]) == 0: image = image[:, :-1]

    rows, cols = image.shape
    if rows > cols:
        pad = (rows - cols) // 2
        image = np.pad(image, ((0, 0), (pad, pad)), mode='constant')
    elif cols > rows:
        pad = (cols - rows) // 2
        image = np.pad(image, ((pad, pad), (0, 0)), mode='constant')

    # Shift center of mass to center of image
    cy, cx = ndimage.center_of_mass(image)
    shiftx = np.round(14 - cx).astype(int)
    shifty = np.round(14 - cy).astype(int)
    M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
    image = cv2.warpAffine(image, M, (28, 28))
    return image

def preprocess(pygame_surface):
    raw_data = pygame.surfarray.array3d(pygame_surface)
    gray = cv2.cvtColor(raw_data, cv2.COLOR_RGB2GRAY)

    # Skip nearly empty canvas
    if np.count_nonzero(gray) < 50:
        return None

    # Preprocessing steps
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)

    # Get contours sorted by area
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(contour)
    margin = 20
    x = max(x - margin, 0)
    y = max(y - margin, 0)
    w = min(w + 2 * margin, thresh.shape[1] - x)
    h = min(h + 2 * margin, thresh.shape[0] - y)
    roi = thresh[y:y + h, x:x + w]

    # Center and resize
    roi = center_digits(roi)
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = roi.astype('float32') / 255.0
    roi = np.expand_dims(roi, axis=-1)
    return np.expand_dims(roi, axis=0)

# Pygame setup
pygame.init()
width, height = 400, 400
white, black = (255, 255, 255), (0, 0, 0)  # Fixed white color (was incorrectly set to yellow)
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Draw a digit (0-9)")
screen.fill(black)
pygame.display.update()
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 36)  # Changed font for better readability

model = get_model()
drawing = False
prediction = None
last_pos = None
start_post = None

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
            
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
            input_image = preprocess(screen)
            if input_image is not None:
                pred = model.predict(input_image)
                prediction = np.argmax(pred)
                
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                screen.fill(black)
                prediction = None
    
    # In your drawing loop:
    if drawing:
        x, y = pygame.mouse.get_pos()
        if last_pos is not None:
            pygame.draw.line(screen, white, last_pos, (x, y), 4)
        pygame.draw.circle(screen, white, (x, y), 4)  # Draw dot regardless
        last_pos = (x, y)
    else:
        last_pos = None  # Reset when not drawing

    if prediction is not None:
        label = font.render(f'Prediction: {prediction}', True, white)
        screen.blit(label, (10, 10))
        
    pygame.display.update()
    clock.tick(60)
    
pygame.quit()