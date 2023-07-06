def find_borders(mask, voxel_dims, width=5):
    struct_elem = np.ones([int(np.clip(width / voxel_dims[i], 2, None)) for i in range(len(mask.shape))])

    ### Locate pixels around lesion boundaries
    outer_border = ndi.binary_dilation(mask, struct_elem).astype(int) - mask
    inner_border = mask - ndi.binary_erosion(mask, struct_elem).astype(int)
    total_border = (outer_border + inner_border > 0).astype(np.uint8)
    euclidian_distance = ndi.distance_transform_edt(1 - total_border)

    ### Generate a weightmap putting weights on pixels close to boundaries
    weight_les = 0.5
    border_weight_param = 0.75
    mask_weight = 0.25
    distance_scaler = -0.09
    boundary_weightmap_lesion = border_weight_param * np.clip(np.exp(distance_scaler * euclidian_distance), weight_les, None) + mask_weight * mask

    return boundary_weightmap_lesion.astype(np.float16)

def main(opt):
    if not os.path.exists(opt.save_path_4_training_slices): os.makedirs(opt.save_path_4_training_slices)

    assign_file_v = CSVlogger(opt.save_path_4_training_slices + "/Assign_2D_Volumes.csv", ["Volume", "Slice Path"])
    assign_file_les = CSVlogger(opt.save_path_4_training_slices + "/Assign_2D_LesionMasks.csv", ["Volume", "Slice Path", "Has Mask"])
    assign_file_wles = CSVlogger(opt.save_path_4_training_slices + "/Assign_2D_LesionWmaps.csv", ["Volume", "Slice Path"])

    volumes  = os.listdir(opt.path_2_training_volumes)
    segs, vols = [],[]
    
    for x in volumes:
        if 'segmentation' in x: segs.append(x)
        if 'volume' in x: vols.append(x)

    vols.sort()
    segs.sort()

    if not os.path.exists(opt.save_path_4_training_slices):
        os.makedirs(opt.save_path_4_training_slices)

    volume_iterator = tqdm(zip(vols, segs), position=0, total=len(vols))    

    for i,data_tuple in enumerate(volume_iterator):
        ### ASSIGNING RELEVANT VARIABLES
        vol, seg = data_tuple

        ### LOAD VOLUME AND MASK DATA
        volume_iterator.set_description('Loading Data...')

        volume = nib.load(opt.path_2_training_volumes + "/" + vol)
        v_name = vol.split(".")[0]
        voxel_dims = volume.header.structarr['pixdim'][1:4][[2,0,1]]

        volume = np.array(volume.dataobj)
        volume = volume.transpose(2,0,1)
        save_path_v = opt.save_path_4_training_slices + "/Volumes/" + v_name
        if not os.path.exists(save_path_v): os.makedirs(save_path_v)
        
        segmentation = np.array(nib.load(opt.path_2_training_volumes + "/" + seg).dataobj)
        segmentation = segmentation.transpose(2,0,1)

        save_path_lesion_masks = opt.save_path_4_training_slices + "/LesionMasks/" + v_name
        save_path_lesion_weightmaps = opt.save_path_4_training_slices + "/BoundaryMasksLesion/" + v_name

        if not os.path.exists(save_path_lesion_masks): os.makedirs(save_path_lesion_masks)
        if not os.path.exists(save_path_lesion_weightmaps): os.makedirs(save_path_lesion_weightmaps)

        volume_iterator.set_description('Generating Masks...')
        lesion_mask = segmentation == 2
        volume_iterator.set_description('Generating Weightmaps...')
        weightmap_lesion = find_borders(lesion_mask, voxel_dims)
        volume_slice_iterator = tqdm(zip(volume, lesion_mask.astype(np.uint8), weightmap_lesion), position=1, total=len(volume))

        volume_iterator.set_description('Saving Slices...')

        for idx,data_tuple in enumerate(volume_slice_iterator):
            (v_slice, les_slice, lesmap_slice) = data_tuple


            np.save(save_path_lesion_weightmaps+"/slice-"+str(idx)+".npy", lesmap_slice)
            np.save(save_path_lesion_masks+"/slice-"+str(idx)+".npy", les_slice.astype(np.uint8))

            assign_file_wles.write([v_name, save_path_lesion_weightmaps+"/slice-"+str(idx)+".npy"])
            assign_file_les.write([v_name, save_path_lesion_masks+"/slice-"+str(idx)+".npy", 1 in les_slice.astype(np.uint8)])

            np.save(save_path_v+"/slice-"+str(idx)+".npy", v_slice.astype(np.int16))
            assign_file_v.write([v_name, save_path_v+"/slice-"+str(idx)+".npy"])

if __name__ == '__main__':

    """=================================="""
    ### LOAD BASIC LIBRARIES
    import numpy as np, os, nibabel as nib, argparse, pickle as pkl
    from dataset_preparation.csv_logger import CSVlogger
    from tqdm import tqdm
    import scipy.ndimage as ndi


    """===================================="""
    ### GET PATHS
    parse_in = argparse.ArgumentParser()
    parse_in.add_argument('--path_2_training_volumes', type=str, default='/home/tvlad/Downloads/original_data_LiTS/Training_Data/',
                          help='Path to original LiTS-volumes in nii-format.')
    parse_in.add_argument('--save_path_4_training_slices', type=str, default='/home/tvlad/Downloads/training_data_LiTS/',
                          help='Where to save the 2D-conversion.')
    
    opt = parse_in.parse_args()

    ### RUN GENERATION
    main(opt)
