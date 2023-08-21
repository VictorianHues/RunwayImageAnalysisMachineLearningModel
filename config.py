import os

DATASET_PATH = r"C:\Users\Victo\Documents\Productive\Projects\RunwayFashionML\dataset"
OUTPUT = r"C:\Users\Victo\Documents\Productive\Projects\RunwayFashionML\output"

IMAGES_PATH = os.path.sep.join([BASE_PATH, r"processedImages"])
ANNOTATION_PATH = os.path.sep.join([BASE_PATH, r"airplanes.csv"])

MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.h5"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_FILENAMES = os.path.sep.join([BASE_OUTPUT, "test_images.txt"])

INITIAL_LEARNING_RATE = 0.0001
NUM_EPOCHS = 25
BATCH_SIZE = 32

