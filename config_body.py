import os

BASE_PATH = os.getcwd()

DATASET_PATH = os.path.sep.join([BASE_PATH, r"dataset"])
OUTPUT = os.path.sep.join([BASE_PATH, r"output"])

IMAGES_PATH = os.path.sep.join([DATASET_PATH, r"images"])
ANNOTATION_PATH = os.path.sep.join([DATASET_PATH, r"annotations"])

MODEL_PATH = os.path.sep.join([OUTPUT, "body_detector.keras"])
PLOT_PATH = os.path.sep.join([OUTPUT, "plot.png"])
TEST_FILENAMES = os.path.sep.join([OUTPUT, "test_images.txt"])

INITIAL_LEARNING_RATE = 0.0001
NUM_EPOCHS = 25
BATCH_SIZE = 32

