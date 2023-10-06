from config_body import *
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split

import numpy as np
import cv2
import os
import json


# Initialize the set of image data, bounding box coordinates, and the set of filenames for individual images
data = []
targets = []
filenames = []


# load the contents of the YOLO formatted annotations files
print("[INFO] loading dataset...")
imageFileSet = [f for f in os.listdir(IMAGES_PATH) if f.endswith((".jpg", ".jpeg", ".png"))] # List all image files in the image directory

for imageFile in imageFileSet: # Loop through image files
    
    # Construct the full image path
    imagePath = os.path.join(IMAGES_PATH, imageFile)

    # Load and preprocess the image
    image = cv2.imread(imagePath)
    (h, w) = image.shape[:2]

    # Create a YOLO format annotation file for the current image
    annotationFilename = os.path.splitext(imageFile)[0] + ".txt"
    annotationPath = os.path.join(ANNOTATION_PATH, annotationFilename)

    # Split the annotation line into individual values
    # !Assuming single line txt file!
    annotationFile = open(annotationPath, 'r')
    annotationLines = annotationFile.readlines()
    for line in annotationLines:
        annotationValues = line.split()
        
        # Parse the values 
        classId = int(annotationValues[0])
        centerX = float(annotationValues[1])
        centerY = float(annotationValues[2])
        width = float(annotationValues[3])
        height = float(annotationValues[4])

        # Normalize coordinates
        startX = float(centerX) / w
        startY = float(centerY) / h
        endX = float(width) / w
        endY = float(height) / h

        # load the image and preprocess it
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
        
        # update our list of data, targets, and filenames
        data.append(image)
        targets.append((startX, startY, endX, endY))
        filenames.append(imageFile)
        # print(imageFile)


# Convert the data and targets to NumPy arrays, scaling the input
# Pixel range [0, 255] -> [0, 1]
data = np.array(data, dtype="float32") / 255.0
targets = np.array(targets, dtype="float32")

# Partition 90% of the data for training and the remaining 10% for testing
split = train_test_split(data, targets, filenames, test_size=0.10, random_state=42)

# Unpack the data split
(trainImages, testImages) = split[:2]
(trainTargets, testTargets) = split[2:4]
(trainFilenames, testFilenames) = split[4:]

# Write the testing filenames to disk to be used when evaluating/testing bounding box regressor
print("[INFO] saving testing filenames...")
f = open(TEST_FILENAMES, "w")
f.write("\n".join(testFilenames))
f.close()


# load the VGG16 network, ensuring the head FC layers are left off
vgg = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# Freeze all VGG layers so they will not be updated during the training process
vgg.trainable = False

# Flatten the max-pooling output of VGG
flatten = vgg.output
flatten = Flatten()(flatten)

# Construct a fully-connected layer header to output the predicted bounding box coordinates
bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid")(bboxHead)

# Construct the model for bounding box regression
model = Model(inputs=vgg.input, outputs=bboxHead)

# Initialize the optimizer, compile the model, and show the model summary
opt = Adam(learning_rate=INITIAL_LEARNING_RATE)
model.compile(loss="mse", optimizer=opt)
print(model.summary())

# Train the network for bounding box regression
print("[INFO] training bounding box regressor...")
H = model.fit(
	trainImages, 
    trainTargets,
	validation_data=(testImages, testTargets),
	batch_size=BATCH_SIZE,
	epochs=NUM_EPOCHS,
	verbose=1)

# Save the model to disk
print("[INFO] saving object detector model...")
model.save(MODEL_PATH, save_format="h5")

# Plot the model training history
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config_body.PLOT_PATH)