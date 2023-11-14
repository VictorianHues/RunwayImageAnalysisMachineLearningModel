from config_body import *
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import mimetypes
import argparse
import imutils
import cv2
import os

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk


def close_window(root):
    root.destroy()


def commandLineSelectImages():
    # Parse arguments
    ap = argparse.ArgumentParser(description="Test Image input")
    ap.add_argument("-i", "--input", required=True, help=r"path to input image/text file of image filenames")
    args = vars(ap.parse_args()) 

    # Determine single input file type
    filetype = mimetypes.guess_type(args["input"])[0]
    imagePaths = [args["input"]]
    
    # If the file type is a text file, then we need to process multiple images
    if "text/plain" == filetype:
    	# load the filenames in our testing file and initialize our list
    	# of image paths
    	filenames = open(args["input"]).read().strip().split("\n")
    	imagePaths = []
    	# loop over the filenames
    	for f in filenames:
    		# construct the full path to the image filename and then
    		# update our image paths list
    		p = os.path.sep.join([config.IMAGES_PATH, f])
    		imagePaths.append(p)

    return imagePaths





def guiSelectImages():
    global guiImagePaths
    
    filePaths = filedialog.askopenfilenames(
        title="Select Image Files",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")],
    )
    if filePaths:
        guiImagePaths = list(filePaths)
        for filePath in filePaths:
            # Display the selected images (you can customize this part)
            img = Image.open(filePath)
            img.thumbnail((100, 100))
            img = ImageTk.PhotoImage(img)
            label = tk.Label(image=img)
            label.image = img
            label.pack()


def selectCompatibleDisplayMethod():
    # Check if a display is available
    try:
        root = tk.Tk()
        root.destroy()
        graphicalDisplayAvailable = True
    except tk.TclError:
        graphicalDisplayAvailable = False

    # print(graphicalDisplayAvailable)

    if graphicalDisplayAvailable:
        print("Initializing GUI file selection")
        
        # A graphical display is available, so use the GUI image selection method
        root = tk.Tk()
        root.title("Image Selector")
        
        selectButton = tk.Button(root, text="Select Images", command=guiSelectImages)
        selectButton.pack()

        closeButton = tk.Button(root, text="Complete Selection", command=lambda: close_window(root))
        closeButton.pack()
    
        root.mainloop()
        imagePaths = guiImagePaths
        # print("Selected image paths:", guiImagePaths)
    else:
        print("Initializing command line file selection")
        imagePaths = commandLineSelectImages()

    print("Selected Images: ", imagePaths)

    return imagePaths


def commandLineSelectModel():
    # Parse arguments
    ap = argparse.ArgumentParser(description="Model Selection:")
    ap.add_argument("-i", "--input", required=True, help=r"path to model file")
    args = vars(ap.parse_args()) 

    filetype = mimetypes.guess_type(args["input"])[0]
    modelPath = [args["input"]]
    

    return modelPath



def guiModelPath():
    global guiModelPath
    
    filePath = filedialog.askopenfilename(
        title="Select Model File",
        filetypes=[("Model", "*.keras *.h5")],
    )
    if filePath:
        guiModelPath = filePath



def selectCompatibleDisplayMethodModel():
    # Check if a display is available
    try:
        root = tk.Tk()
        root.destroy()
        graphicalDisplayAvailable = True
    except tk.TclError:
        graphicalDisplayAvailable = False

    # print(graphicalDisplayAvailable)

    if graphicalDisplayAvailable:
        print("Initializing GUI file selection")

        root = tk.Tk()
        root.title("Model Selector")
        
        selectButton = tk.Button(root, text="Select File", command=guiModelPath)
        selectButton.pack()

        closeButton = tk.Button(root, text="Complete Selection", command=lambda: close_window(root))
        closeButton.pack()
    
        root.mainloop()
        modelPath = guiModelPath
    else:
        print("Initializing command line keras file selection")
        modelPath = commandLineSelectModel()

    print("Selected model: ", modelPath)

    return modelPath




if __name__ == "__main__":
    imagePaths = selectCompatibleDisplayMethod()

    # load our trained bounding box regressor from disk
    print("[INFO] loading object detector...")

    modelPath = selectCompatibleDisplayMethodModel()
    ##print("Model loaded: ", MODEL_PATH)
    ##model = load_model(MODEL_PATH)

    print("Model loaded: ", modelPath)

    model = load_model(modelPath)
    model.summary()

    # loop over the images using the bounding box regression model
    for imagePath in imagePaths:
    	# load the input image in Keras format and scale pixel intensities to the range [0, 1]
    	image = load_img(imagePath, target_size=(224, 224))
    	image = img_to_array(image) / 255.0
    	image = np.expand_dims(image, axis=0)
    
        # Make bounding box predictions on the input image
    	preds = model.predict(image)[0]
    	(startX, startY, endX, endY) = preds
    	
        # load the input image in OpenCV format, resize it such that it fits on our screen, and grab its dimensions
    	image = cv2.imread(imagePath)
    	image = imutils.resize(image, width=600)
    	(h, w) = image.shape[:2]
        
    	# Scale the predicted bounding box coordinates based on the image dimensions
    	startX = int(startX * w)
    	startY = int(startY * h)
    	endX = int(endX * w)
    	endY = int(endY * h)
        
    	# Draw the predicted bounding box on the image
    	cv2.rectangle(image, (startX, startY), (endX, endY),
    		(0, 255, 0), 2)
        
    	# Show the output image
    	cv2.imshow("Output", image)
    	cv2.waitKey(0)
