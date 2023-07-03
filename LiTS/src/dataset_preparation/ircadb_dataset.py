import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from model_training import preprocessing_utils

class IRCADB_Dataset(Dataset):
    def __init__(self, root_path):
        VOLUMES_SLICES_LEDGER_NAME = "2D_Volumes_Positive.csv"
        VOLUMES_MASKS_LEDGER_NAME = "2D_LesionMasks_Positive.csv"
        self.volume_slices = pd.read_csv(os.path.join(root_path, VOLUMES_SLICES_LEDGER_NAME))
        self.volume_masks = pd.read_csv(os.path.join(root_path, VOLUMES_MASKS_LEDGER_NAME))

    def __getitem__(self, index):
        # Read slice and corresponding mask
        slice_entry, mask_entry = self.volume_slices.iloc[index], self.volume_masks.iloc[index]
        
        if slice_entry["Volume"] != mask_entry["Volume"] or slice_entry["Slice Path"].split("/")[-1] != mask_entry["Slice Path"].split("/")[-1]:
            raise Exception(f"Found incompatible entries in annotation files for index {index}")
        
        current_volume = slice_entry["Volume"]

        if index + 1 < len(self.volume_slices):
            next_slice_entry, next_mask_entry = self.volume_slices.iloc[index + 1], self.volume_masks.iloc[index + 1]
            if next_slice_entry["Volume"] != next_mask_entry["Volume"] or next_slice_entry["Slice Path"].split("/")[-1] != next_mask_entry["Slice Path"].split("/")[-1]:
                raise Exception(f"Found incompatible entries in annotation files for index {index + 1}")
            next_volume = next_slice_entry["Volume"]
        else:
            next_volume = current_volume

        # Preprocess slice and corresponding mask
        slice_data = preprocessing_utils.normalize(preprocessing_utils.reorient_to_match_training(np.load(slice_entry["Slice Path"])))
        slice_data = np.expand_dims(slice_data, 0)

        mask_data = np.load(mask_entry["Slice Path"])
        mask_data = np.expand_dims(preprocessing_utils.reorient_to_match_training(mask_data), 0)

        # Register a change in volume
        volume_change = current_volume != next_volume

        return {
            "input_image": (slice_data).astype(np.float32),
            "input_mask": (mask_data).astype(np.float32),
            "volume": current_volume,
            "volume_change": volume_change,
            "slice_path": slice_entry["Slice Path"],
            "mask_path": mask_entry["Slice Path"]
        }

    def __len__(self):
        if len(self.volume_slices) == len(self.volume_masks):
            return len(self.volume_slices)
        else:
            raise Exception(f"Expected number of entries in slices and masks ledgers to be identical. Found slices: {len(self.volume_slices)}, masks: {len(self.volume_masks)}")
