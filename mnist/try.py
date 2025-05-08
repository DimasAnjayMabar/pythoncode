import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Show dataset shapes
print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

# Display the first training image
plt.imshow(x_train[0], cmap='gray')  # use cmap='gray' for proper grayscale display
plt.title(f"Label: {y_train[0]}")
plt.axis('off')
plt.show()

