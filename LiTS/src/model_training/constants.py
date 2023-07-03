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
ACADTUM_DATASET_MEAN = 0.1085

# ACADTUM Pixel std after normalization over full dataset
ACADTUM_DATASET_STD = 0.16864

ROOT_PREPROCESSED_TRAINING_DATA_PATH = "/home/tvlad/Downloads/training_data_LiTS"

EPSILON = 1e-6