import os
import numpy as np
from torch.utils.data import Dataset
from model_training import preprocessing_utils
import pandas as pd

class LiTSDataset(Dataset):
    """
    Summary:
    ----
    Dataset class to retrieve samples from the the LiTS dataset one sample at the time
    """

    def __init__(self, volume_info, scan_volumes, config, is_validation=False) -> None:
        self.config = config
        self.channel_size = 1
        self.volume_info = volume_info
        self.is_validation = is_validation
        self.available_volumes = scan_volumes
        self.rng = np.random.RandomState(self.config["seed"])
        self.rvic = self.config["Training_ROI_Vicinity"]


        self.input_samples = {
            'Neg':[],
            'Pos':[]
            }

        self.div_in_volumes = {
            key: {
                    'Input_Image_Paths':[],
                    'Has Target Mask':[],
                    'Wmap_Paths':[],
                    'TargetMask_Paths':[],
                    'Has Ref Mask':[],
                    'RefMask_Paths':[]
                } for key in self.available_volumes
            }
        
        # Record data paths for each slice in all training/validation volumes:
        # * scan slice
        # * object annotation flag (whether or not object class is visible in the scan slice/annotated scan slice)
        # * weightmap
        # * scan annotation
        # * liver annotation flag (whether or not the liver is visible in the scan slice/annotated scan slice)
        # * annotated liver scan slice
        for i,vol in enumerate(volume_info['volume_slice_info']['Volume']):
            if vol in self.div_in_volumes.keys():
                self.div_in_volumes[vol]['Input_Image_Paths'].append(volume_info['volume_slice_info']['Slice Path'][i])
                self.div_in_volumes[vol]['Has Target Mask'].append(volume_info['target_mask_info']['Has Mask'][i])
                if self.config['use_weightmaps']: self.div_in_volumes[vol]['Wmap_Paths'].append(volume_info['weight_mask_info']['Slice Path'][i])
                self.div_in_volumes[vol]['TargetMask_Paths'].append(volume_info['target_mask_info']['Slice Path'][i])
                if self.config['data']=='lesion': self.div_in_volumes[vol]['Has Ref Mask'].append(volume_info['ref_mask_info']['Has Mask'][i])
                if self.config['data']=='lesion': self.div_in_volumes[vol]['RefMask_Paths'].append(volume_info['ref_mask_info']['Slice Path'][i])

        self.volume_details = {
            key: {
                    'Input_Image_Paths':[],
                    'TargetMask_Paths':[],
                    'Wmap_Paths':[],
                    'RefMask_Paths':[]
                } for key in self.available_volumes
            }

        # Populate dictionary with all necessary data for training
        for i,vol in enumerate(self.div_in_volumes.keys()):
            for j in range(len(self.div_in_volumes[vol]['Input_Image_Paths'])):
                crop_condition = np.sum(self.div_in_volumes[vol]['Has Ref Mask'][int(np.clip(j-self.rvic, 0, None)):j+self.rvic])
                
                if self.config['data']=='liver':
                    crop_condition = True

                if crop_condition:
                    extra_ch = self.channel_size//2
                    low_bound, low_diff = np.clip(j-extra_ch,0,None).astype(int), extra_ch-j
                    up_bound, up_diff = np.clip(j+extra_ch+1,None,len(self.div_in_volumes[vol]["Input_Image_Paths"])).astype(int), j+extra_ch+1-len(self.div_in_volumes[vol]["Input_Image_Paths"])

                    vol_slices = self.div_in_volumes[vol]["Input_Image_Paths"][low_bound:up_bound]

                    if low_diff > 0:
                        extra_slices = self.div_in_volumes[vol]["Input_Image_Paths"][low_bound+1:low_bound+1+low_diff][::-1]
                        vol_slices = extra_slices + vol_slices
                    
                    if up_diff > 0:
                        extra_slices = self.div_in_volumes[vol]["Input_Image_Paths"][up_bound-up_diff-1:up_bound-1][::-1]
                        vol_slices = vol_slices + extra_slices

                    self.volume_details[vol]['Input_Image_Paths'].append(vol_slices)
                    self.volume_details[vol]['TargetMask_Paths'].append(self.div_in_volumes[vol]['TargetMask_Paths'][j])

                    if self.config['data'] != 'liver':
                        self.volume_details[vol]['RefMask_Paths'].append(self.div_in_volumes[vol]['RefMask_Paths'][j])
                    
                    if self.config['use_weightmaps']:
                        self.volume_details[vol]['Wmap_Paths'].append(self.div_in_volumes[vol]['Wmap_Paths'][j])

                    type_key = 'Pos' if self.div_in_volumes[vol]['Has Target Mask'][j] or self.is_validation else 'Neg'
                    self.input_samples[type_key].append((vol, len(self.volume_details[vol]['Input_Image_Paths']) - 1))

        self.n_files  = np.sum([len(self.input_samples[key]) for key in self.input_samples.keys()])
        self.curr_vol = self.input_samples['Pos'][0][0] if len(self.input_samples['Pos']) else self.input_samples['Neg'][0][0]

    def __getitem__(self, idx) -> None:
        #Choose a positive example with 50% change if training.
        #During validation, 'Pos' will contain all validation samples.
        #Note that again, volumes without lesions/positive target masks need to be taken into account.
        type_choice = not idx % self.config['pos_sample_chance'] or self.is_validation
        modes = list(self.input_samples.keys())
        type_key = modes[type_choice] if len(self.input_samples[modes[type_choice]]) else modes[not type_choice]
    
        type_len = len(self.input_samples[type_key])

        next_vol, _ = self.input_samples[type_key][(idx + 1) % type_len]
        vol, idx = self.input_samples[type_key][idx % type_len]

        vol_change = next_vol != vol
        self.curr_vol = vol
    
        intvol = self.volume_details[vol]["Input_Image_Paths"][idx]
        intvol = intvol[len(intvol) // 2]

        input_image  = np.concatenate([np.expand_dims(np.load(sub_vol),0) for sub_vol in self.volume_details[vol]["Input_Image_Paths"][idx]], axis=0)

        #Perform data standardization
        if self.config['no_standardize']:
            input_image  = preprocessing_utils.normalize(input_image, zero_center=False, unit_variance=False, supply_mode="orig")
        else:
            input_image  = preprocessing_utils.normalize(input_image)

        #Lesion/Liver Mask to output
        target_mask = np.load(self.volume_details[vol]["TargetMask_Paths"][idx])
        target_mask = np.expand_dims(target_mask, 0)

        #Liver Mask to use for defining training region of interest
        crop_mask = np.expand_dims(np.load(self.volume_details[vol]["RefMask_Paths"][idx]),0) if self.config['data']=='lesion' else None
        
        #Weightmap to output
        weightmap = np.expand_dims(np.load(self.volume_details[vol]["Wmap_Paths"][idx]),0).astype(float) if self.config['use_weightmaps'] else None

        #Generate list of all files that would need to be cropped, if cropping is required.
        files_to_crop  = [input_image, target_mask]
        is_mask        = [0,1]
        if weightmap is not None:
            files_to_crop.append(weightmap)
            is_mask.append(0)
        if crop_mask is not None:
            files_to_crop.append(crop_mask)
            is_mask.append(1)

        #First however, augmentation, if required, is performed (i.e. on fullsize images to remove border artefacts in crops).
        if len(self.config["augment"]) and not self.is_validation:
            files_to_crop = list(preprocessing_utils.augment_2D(files_to_crop, mode_dict = self.config["augment"],
                                               seed=self.rng.randint(0,1e8), is_mask = is_mask))

        #If Cropping is required, we crop now.
        if len(self.config['crop_size']) and not self.is_validation:
            #Add imaginary batch axis in gu.get_crops_per_batch
            crops_for_picked_batch  = preprocessing_utils.get_crops_per_batch(files_to_crop, crop_mask, crop_size=self.config['crop_size'], seed=self.rng.randint(0,1e8))
            input_image     = crops_for_picked_batch[0]
            target_mask     = crops_for_picked_batch[1]
            weightmap       = crops_for_picked_batch[2] if weightmap is not None else None
            crop_mask       = crops_for_picked_batch[-1] if crop_mask is not None else None

        one_hot_target = preprocessing_utils.numpy_generate_onehot_matrix(target_mask, self.config['num_classes']) if self.config['require_one_hot'] else None

        # Convert data to polar coordinates if necessary
        if self.config["polar_training"]:
            # Compute polar origin based on annotated region of interest centroid
            roi_centroid = preprocessing_utils.centroid(target_mask[0])

            if np.random.uniform() < 0.3:
                center_max_shift = 0.05 * 128
                roi_centroid = np.array(roi_centroid)
                roi_centroid = (
                    roi_centroid[0] + np.random.uniform(-center_max_shift, center_max_shift),
                    roi_centroid[1] + np.random.uniform(-center_max_shift, center_max_shift)
                )

            # Transform to polar cooridnates
            input_image = np.expand_dims(preprocessing_utils.to_polar(input_image[0], roi_centroid), 0)
            target_mask = np.expand_dims(preprocessing_utils.to_polar(target_mask[0], roi_centroid), 0)
            crop_mask = np.expand_dims(preprocessing_utils.to_polar(crop_mask[0], roi_centroid), 0) if crop_mask is not None else None
            weightmap = np.expand_dims(preprocessing_utils.to_polar(weightmap[0], roi_centroid), 0) if weightmap is not None else None
            if one_hot_target is not None:
                converted_to_polar = []
                for ch in range(one_hot_target.shape[0]):
                    converted_to_polar.append(preprocessing_utils.to_polar(one_hot_target[ch], roi_centroid))
                one_hot_target = np.stack(converted_to_polar, axis=0)
        
        #Final Output Dictionary
        return_dict = {
            "input_images":input_image.astype(np.float32),
            "targets":target_mask.astype(np.float32),
            "crop_option":crop_mask.astype(np.float32) if crop_mask is not None else None,
            "weightmaps":weightmap.astype(np.float32) if weightmap is not None else None,
            "one_hot_targets": one_hot_target.astype(np.float32),
            'internal_slice_name':intvol,
            'vol_change':vol_change
        }

        return_dict = {
            key:item for key,item in return_dict.items() if item is not None
        }
        return return_dict

    def __len__(self) -> int:
        return self.n_files
        
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
        slice_data = preprocessing_utils.normalize(preprocessing_utils.reorient_to_match_training(np.load(slice_entry["Slice Path"])), zero_center=False, unit_variance=False)
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

class ACADTUM_Dataset(Dataset):
    def __init__(self, root_path):
        VOLUMES_SLICES_LEDGER_NAME = "2D_Volumes_Positive.csv"
        VOLUMES_MASKS_LEDGER_NAME = "2D_LesionMasks_Positive.csv"
        self.volume_slices = pd.read_csv(os.path.join(root_path, VOLUMES_SLICES_LEDGER_NAME))
        self.volume_masks = pd.read_csv(os.path.join(root_path, VOLUMES_MASKS_LEDGER_NAME))

    def __getitem__(self, index):
        # Read slice and corresponding mask
        slice_entry, mask_entry = self.volume_slices.iloc[index], self.volume_masks.iloc[index]
        
        if slice_entry["Volume"].split("[")[0] != mask_entry["Volume"].split("[")[0] or slice_entry["Slice Path"].split("/")[-1] != mask_entry["Slice Path"].split("/")[-1]:
            raise Exception(f"Found incompatible entries in annotation files for index {index}")
        
        current_volume = slice_entry["Volume"].split("[")[0]

        if index + 1 < len(self.volume_slices):
            next_slice_entry, next_mask_entry = self.volume_slices.iloc[index + 1], self.volume_masks.iloc[index + 1]
            if next_slice_entry["Volume"].split("[")[0] != next_mask_entry["Volume"].split("[")[0] or next_slice_entry["Slice Path"].split("/")[-1] != next_mask_entry["Slice Path"].split("/")[-1]:
                raise Exception(f"Found incompatible entries in annotation files for index {index + 1}")
            next_volume = next_slice_entry["Volume"].split("[")[0]
        else:
            next_volume = current_volume

        # Preprocess slice and corresponding mask
        slice_data = preprocessing_utils.normalize((np.load(slice_entry["Slice Path"])), zero_center=False, unit_variance=False)
        slice_data = np.expand_dims(slice_data, 0)

        mask_data = np.load(mask_entry["Slice Path"])
        mask_data = np.expand_dims((mask_data), 0)

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
        