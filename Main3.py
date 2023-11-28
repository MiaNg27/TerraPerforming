import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from PIL import Image

import matplotlib.pyplot as plt
import tensorflow as tf
import keras.api._v2.keras as keras
from tensorflow.python.keras import layers
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

import cv2
import numpy as np


setMudras = tf.TensorArray(dtype=tf.float16, size=700, element_shape=(128, 128,3))
mudras7 = np.empty(7, dtype=np.ndarray)
for i in range(7):
    mudras7[i] = np.empty((128, 128,3),dtype=np.uint8)
for i in range(1, 8):
    image = cv2.imread(f'image{i}.jpg')

    # Resize the image
    width, height = 128, 128
    image = cv2.resize(image, (width, height))
    mudras7[i-1] = image

    # Convert the image to grayscale (optional)
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image_array = np.array(gray_image)
    for j in range(1, 101):
        # Add each image as a tensor to the setMudras TensorArray
        setMudras = setMudras.write(j, image)
# Convert the TensorArray to a stacked tensor
setMudras = setMudras.stack()


excavator = cv2.VideoCapture('excavator.mp4')
if not excavator.isOpened():
    print("Error: Could not open video file.")
    exit()
trainData = tf.TensorArray(dtype=tf.float16, size=700, element_shape=(128, 128,3))
a = 0
while True:
    ret, frame = excavator.read()

    if not ret:
        break
    frame_array = cv2.resize(frame, (width, height))
    #gray_frame = cv2.cvtColor(frame_array, cv2.COLOR_BGR2GRAY)
    #frame_array = gray_frame

    if a < 700:
        # Add each frame_array as a tensor to the trainData TensorArray
        trainData = trainData.write(a, frame_array)
        a += 1

# Convert the TensorArray to a stacked tensor
trainData = trainData.stack()

X = tf.convert_to_tensor(trainData)
y = tf.convert_to_tensor(setMudras)


total_samples = trainData.shape[0]
train_size = int(0.7 * total_samples)
test_size = int(0.2 * total_samples)
val_size = total_samples - train_size - test_size

# Split the tensors into training, testing, and validation sets
X_train = trainData[:train_size]
X_test = trainData[train_size:train_size + test_size]
X_val = trainData[train_size + test_size:]

y_train = setMudras[:train_size]
y_test = setMudras[train_size:train_size + test_size]
y_val = setMudras[train_size + test_size:]


# Define U-Net model
def unet_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Encoder
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D()(conv1)

    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D()(conv2)

    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D()(conv3)

    # Bottleneck
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)

    # Decoder
    up1 = layers.UpSampling2D()(conv4)
    concat1 = layers.concatenate([conv3, up1], axis=-1)
    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(concat1)
    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv5)

    up2 = layers.UpSampling2D()(conv5)
    concat2 = layers.concatenate([conv2, up2], axis=-1)
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(concat2)
    conv6 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv6)

    up3 = layers.UpSampling2D()(conv6)
    concat3 = layers.concatenate([conv1, up3], axis=-1)
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(concat3)
    conv7 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv7)

    # Output layer
    outputs = layers.Conv2D(num_classes, 1, activation='softmax')(conv7)

    model = keras.Model(inputs, outputs)

    return model

# Define input shape and number of classes
input_shape = (128, 128  , 3)
num_classes = 3  # Adjust as per your task

# Create and compile the U-Net model
model = unet_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3 )  # Adjust the number of epochs
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(test_loss)
print(test_accuracy)
################################################################


drill = cv2.VideoCapture('Drill.mp4')
if not excavator.isOpened():
    print("Error: Could not open video file.")
    exit()
drillData = tf.TensorArray(dtype=tf.float16, size=700, element_shape=(128, 128,3))
a = 0
while True:
    ret, frame = drill.read()

    if not ret:
        break
    frame_array = cv2.resize(frame, (width, height))
    #gray_frame = cv2.cvtColor(frame_array, cv2.COLOR_BGR2GRAY)
    #frame_array = gray_frame

    if a < 700:
        # Add each frame_array as a tensor to the trainData TensorArray
        drillData = drillData.write(a, frame_array)
        a += 1

# Convert the TensorArray to a stacked tensor
drillData = drillData.stack()

result = model.predict(trainData)
resultDrill = model.predict(drillData)


#################################################################

numpyMudras = []
for i in range(len(setMudras)):  # Get the size of the TensorArray
    numpy_element = setMudras[i].numpy()
    numpyMudras.append(numpy_element)

distances = []

setResults = np.empty(7, dtype=np.ndarray)
images = []
for a in [result,resultDrill]:
    for i in range(0,len(a),200):
        average_matrix = np.mean(a[i:i+200], axis=0)
        for j in range(len(mudras7)):
            if mudras7[j] is not None:
                distance = np.linalg.norm(average_matrix - mudras7[j])
                distances.append(distance)
                closest_index = np.argmin(distances)
                if closest_index < len(mudras7):
                    images.append(mudras7[closest_index])

excavator_images = images[:10]
mudras7_list = list(mudras7)+ list(mudras7)
drill_images = random.sample(mudras7_list, 10)


# Create a figure and specify the number of rows and columns for subplots
num_rows = 2
num_cols = 5
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5))  # You can adjust the figure size
fig1,axes1 = plt.subplots(num_rows, num_cols, figsize=(15, 5))
fig2,axes2 = plt.subplots(num_rows, num_cols, figsize=(15, 5))
fig3,axes3 = plt.subplots(num_rows, num_cols, figsize=(15, 5))

# Assuming 'selected_images' is a list of NumPy arrays
for i, ax in enumerate(axes.flat):  # Use 'axes.flat' to iterate over subplots
    a = Image.fromarray(excavator_images[i])
    ax.imshow(a, cmap='gray')  # Change 'cmap' to 'viridis' or other colormaps as needed
    ax.set_title(f"Image {i + 1}")
    ax.axis('off')

for i, ax in enumerate(axes1.flat):  # Use 'axes.flat' to iterate over subplots
    a = Image.fromarray(drill_images[i])
    ax.imshow(a, cmap='gray')  # Change 'cmap' to 'viridis' or other colormaps as needed
    ax.set_title(f"Image {i + 1}")
    ax.axis('off')

for i, ax in enumerate(axes2.flat):  # Use 'axes.flat' to iterate over subplots
    a = Image.fromarray(random.sample(mudras7_list, 10)[i])
    ax.imshow(a, cmap='gray')  # Change 'cmap' to 'viridis' or other colormaps as needed
    ax.set_title(f"Image {i + 1}")
    ax.axis('off')

for i, ax in enumerate(axes3.flat):  # Use 'axes.flat' to iterate over subplots
    a = Image.fromarray(random.sample(mudras7_list, 10)[i])
    ax.imshow(a, cmap='gray')  # Change 'cmap' to 'viridis' or other colormaps as needed
    ax.set_title(f"Image {i + 1}")
    ax.axis('off')

# Display the plot



space = ["D","I"]
time = ["Q","S"]
weight = ["H","L"]
flow = ["B","F"]

punch = ["Punch : DQHS"]
dab = ["Dab: DQLB"]
press = ["Press : DSHB"]
glide = ["Glide : DSLF"]
slash = ["Slash: IQHF"]
flick = ["Flick : IQLF"]
wring = ["Wring : ISHB"]
fload = ["Fload : ISLF"]

def generate_random_strings():
    # Create a list of the variable names
    variable_names = [punch, dab, press, glide, slash, flick, wring, fload]

    # Choose two random variables
    random_variables = random.sample(variable_names, 2)

    # Print the chosen variables as strings
    for var in random_variables:
        return  (f"{random_variables[0]}, {random_variables[1]}")

# Call the function to generate and print random strings

fig.suptitle(f"Elements : {generate_random_strings()}",fontsize=20, color='blue')
fig1.suptitle(f"Elements : {generate_random_strings()}",fontsize=20, color='blue')
fig2.suptitle(f"Elements : {generate_random_strings()}",fontsize=20, color='blue')
fig3.suptitle(f"Elements : {generate_random_strings()}",fontsize=20, color='blue')
plt.tight_layout()
plt.show()

