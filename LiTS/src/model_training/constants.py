# Minimum Hounsfield voxel value for tissue. Any value smaller than this is conventinally thought to correspond to water.
MIN_BOUND = -100

# Maximum Hounsfield voxel value for tissue. Any value greater than this is conventionally thought to correspond to bone structure.
MAX_BOUND = 400

# TODO: Figure out how the mean and std values are computed

# LiTS Pixel mean after normalization over full dataset
LITS_DATASET_MEAN = 0.1021

# LiTS Pixel std after normalization over full dataset
LITS_DATASET_STD = 0.19177

# ACADTUM Pixel mean after normalization over full dataset
LITS_DATASET_MEAN = 0.1085

# ACADTUM Pixel std after normalization over full dataset
LITS_DATASET_STD = 0.16864

ROOT_PREPROCESSED_TRAINING_DATA_PATH = "../unet-lits-2d-pipeline/LOADDATA/Training_Data_2D/"
ROOT_PREPROCESSED_TEST_DATA_PATH = "../unet-lits-2d-pipeline/LOADDATA/Test_Data_2D/"

ALPHA_VALUE = 0.05

EPSILON = 1e-6