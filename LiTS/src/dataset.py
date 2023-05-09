import numpy as np
from torch.utils.data import Dataset
import preprocessing_utils

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

        self.data_augmentation = True
        if len(self.config["augment"]) == 0 and self.is_validation:
            self.data_augmentation = False

        self.input_samples = {
            'Neg':[], 'Pos':[]
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
                if self.config['data']=='liver': crop_condition=True

                if crop_condition:
                    extra_ch = self.channel_size//2
                    low_bound, low_diff = np.clip(j-extra_ch,0,None).astype(int), extra_ch-j
                    up_bound, up_diff = np.clip(j+extra_ch+1,None,len(self.div_in_volumes[vol]["Input_Image_Paths"])).astype(int), j+extra_ch+1-len(self.div_in_volumes[vol]["Input_Image_Paths"])

                    vol_slices = self.div_in_volumes[vol]["Input_Image_Paths"][low_bound:up_bound]

                    if low_diff>0:
                        extra_slices = self.div_in_volumes[vol]["Input_Image_Paths"][low_bound+1:low_bound+1+low_diff][::-1]
                        vol_slices = extra_slices+vol_slices
                    if up_diff>0:
                        extra_slices = self.div_in_volumes[vol]["Input_Image_Paths"][up_bound-up_diff-1:up_bound-1][::-1]
                        vol_slices = vol_slices+extra_slices

                    self.volume_details[vol]['Input_Image_Paths'].append(vol_slices)
                    self.volume_details[vol]['TargetMask_Paths'].append(self.div_in_volumes[vol]['TargetMask_Paths'][j])

                    if self.config['data']!='liver':  self.volume_details[vol]['RefMask_Paths'].append(self.div_in_volumes[vol]['RefMask_Paths'][j])
                    if self.config['use_weightmaps']: self.volume_details[vol]['Wmap_Paths'].append(self.div_in_volumes[vol]['Wmap_Paths'][j])

                    type_key = 'Pos' if self.div_in_volumes[vol]['Has Target Mask'][j] or self.is_validation else 'Neg'
                    self.input_samples[type_key].append((vol, len(self.volume_details[vol]['Input_Image_Paths'])-1))

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

        vol, idx = self.input_samples[type_key][idx%type_len]
        next_vol, _ = self.input_samples[type_key][(idx+1)%type_len]

        vol_change = next_vol != vol
        self.curr_vol = vol
    
        intvol = self.volume_details[vol]["Input_Image_Paths"][idx]
        intvol = intvol[len(intvol)//2]

        input_image  = np.concatenate([np.expand_dims(np.load(sub_vol),0) for sub_vol in self.volume_details[vol]["Input_Image_Paths"][idx]],axis=0)

        #Perform data standardization
        if self.config['no_standardize']:
            input_image  = preprocessing_utils.normalize(input_image, zero_center=False, unit_variance=False, supply_mode="orig")
        else:
            input_image  = preprocessing_utils.normalize(input_image)

        #Lesion/Liver Mask to output
        target_mask = np.load(self.volume_details[vol]["TargetMask_Paths"][idx])
        target_mask = np.expand_dims(target_mask,0)


        #Liver Mask to use for defining training region of interest
        crop_mask = np.expand_dims(np.load(self.volume_details[vol]["RefMask_Paths"][idx]),0) if self.config['data']=='lesion' else None
        #Weightmask to output
        weightmap = np.expand_dims(np.load(self.volume_details[vol]["Wmap_Paths"][idx]),0).astype(float) if self.config['use_weightmaps'] else None


        #Generate list of all files that would need to be crop, if cropping is required.
        files_to_crop  = [input_image, target_mask]
        is_mask        = [0,1]
        if weightmap is not None:
            files_to_crop.append(weightmap)
            is_mask.append(0)
        if crop_mask is not None:
            files_to_crop.append(crop_mask)
            is_mask.append(1)

        #First however, augmentation, if required, is performed (i.e. on fullsize images to remove border artefacts in crops).
        if self.data_augmentation:
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

        #Final Output Dictionary
        return_dict = {
            "input_images":input_image.astype(float),
            "targets":target_mask.astype(float),
            "crop_option":crop_mask.astype(float) if crop_mask is not None else None,
            "weightmaps":weightmap.astype(float) if weightmap is not None else None,
            'internal_slice_name':intvol,
            'vol_change':vol_change
        }

        return_dict = {
            key:item for key,item in return_dict.items() if item is not None
        }
        return return_dict

    def __len__(self) -> int:
        return self.n_files
        
