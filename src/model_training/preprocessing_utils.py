from model_training import constants
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import cv2 as cv

def set_bounds(image,min_bound,max_bound):
    return np.clip(image, min_bound, max_bound)

def normalize(image, use_bd=True, zero_center=True, unit_variance=True):
    min_bound, max_bound = None, None
    if not use_bd:
        min_bound = np.min(image)
        max_bound = np.max(image)
    else:
        min_bound = float(constants.MIN_BOUND)
        max_bound = float(constants.MAX_BOUND)
        image = set_bounds(image, min_bound, max_bound)

    image = (image - min_bound) / (max_bound - min_bound)
    image = np.clip(image,0.,1.)
    
    if zero_center:
        image = image - constants.LITS_DATASET_MEAN
    if unit_variance:
        image = image / constants.LITS_DATASET_STD
    return image

def rotate_2D(to_aug, rng=np.random.RandomState(1)):
    angle = (rng.rand()*2-1)*10
    for i,aug_file in enumerate(to_aug):
        for ch in range(aug_file.shape[0]):
            #actually perform rotation
            aug_file[ch,:]    = ndi.rotate(aug_file[ch,:].astype(np.float32), angle, reshape=False, order=0, mode="nearest")
    return to_aug, angle

def zoom_2D(to_aug, rng=np.random.RandomState(1)):
    magnif = rng.uniform(0.825,1.175)
    for i,aug_file in enumerate(to_aug):
        for ch in range(aug_file.shape[0]):
            sub_img     = aug_file[ch,:]
            img_shape   = np.array(sub_img.shape)
            new_shape   = [int(np.round(magnif*shape_val)) for shape_val in img_shape]
            zoomed_shape= (magnif,)*(sub_img.ndim)

            if magnif<1:
                how_much_to_clip    = [(x-y)//2 for x,y in zip(img_shape, new_shape)]
                idx_cornerpix       = tuple(-1 for _ in range(sub_img.ndim))
                idx_zoom            = tuple(slice(x,x+y) for x,y in zip(how_much_to_clip,new_shape))
                zoomed_out_img      = np.ones_like(sub_img)*sub_img[idx_cornerpix]
                zoomed_out_img[idx_zoom] = ndi.zoom(sub_img.astype(np.float32),zoomed_shape,order=0,mode="nearest")
                aug_file[ch,:]        = zoomed_out_img

            if magnif>1:
                zoomed_in_img       = ndi.zoom(sub_img.astype(np.float32),zoomed_shape,order=0,mode="nearest")
                rounding_correction = [(x-y)//2 for x,y in zip(zoomed_in_img.shape,img_shape)]
                rc_idx              = tuple(slice(x,x+y) for x,y in zip(rounding_correction, img_shape))
                aug_file[ch,:]   = zoomed_in_img[rc_idx]

    return to_aug

def hflip_2D(to_aug, rng=np.random.RandomState(1)):
    for i,aug_file in enumerate(to_aug):
        for ch in range(aug_file.shape[0]):
            aug_file[ch,:]  = np.fliplr(aug_file[ch,:])

    return to_aug

def vflip_2D(to_aug, rng=np.random.RandomState(1)):
    for i,aug_file in enumerate(to_aug):
        for ch in range(aug_file.shape[0]):
            aug_file[ch,:]  = np.flipud(aug_file[ch,:])

    return to_aug

def augment_2D(to_aug, mode_dict=["rot","zoom"], seed=1):
    rng = np.random.RandomState(seed)
    modes = []

    if rng.randint(2) and "rot" in mode_dict:
        modes.append('rot')
        to_aug, rotation_angle = rotate_2D(to_aug,rng)
    if rng.randint(2) and "zoom" in mode_dict:
        modes.append('zoom')
        to_aug = zoom_2D(to_aug,rng)
    if rng.randint(2) and "hflip" in mode_dict:
        modes.append('hflip')
        to_aug = hflip_2D(to_aug,rng)
    if rng.randint(2) and "vflip" in mode_dict:
        modes.append('vflip')
        to_aug = vflip_2D(to_aug,rng)

    return to_aug

def get_crops_per_batch(batch_to_crop, idx_batch=None, crop_size=[128,128], seed=1):
    rng = np.random.RandomState(seed)

    sup = list(1 - np.array(crop_size)%2)
    bl_len = len(batch_to_crop)
    batch_list_to_return = []

    if idx_batch is not None:
        all_crop_idxs = np.where(idx_batch[0,:]==1) if np.sum(idx_batch[0,:])!=0 else [[]]
    else:
        all_crop_idxs = [[]]

    if len(all_crop_idxs[0]) > 0:
        if idx_batch is not None:
            crop_idx = [np.clip(rng.choice(ax),crop_size[i]//2-1,batch_to_crop[0][:].shape[i+1]-crop_size[i]//2-1) for i,ax in enumerate(all_crop_idxs)]
    else:
        crop_idx = [rng.randint(crop_size[i] // 2 - 1, np.array(batch_to_crop[0].shape[i+1]) - crop_size[i] // 2 - 1) for i in range(batch_to_crop[0].ndim - 1)]
    
    crop_coordinates = [(center - crop_size[i] // 2 + mv, center + crop_size[i] // 2 + 1) for i, (center, mv) in enumerate(zip(list(crop_idx),sup))]
    
    for i in range(bl_len):
        batch_list_to_return.append(batch_to_crop[i][:, crop_coordinates[0][0]: crop_coordinates[0][1], crop_coordinates[1][0]: crop_coordinates[1][1]])

    return tuple(batch_list_to_return)

def numpy_generate_onehot_matrix(matrix_mask, ndim):
    onehot_matrix = np.eye(ndim)[matrix_mask.reshape(-1).astype('int')].astype('int')
    data_shape    = list(matrix_mask.shape)
    data_shape[0] = ndim
    onehot_matrix = np.fliplr(np.flipud(onehot_matrix).T).reshape(*data_shape)
    return onehot_matrix

def centroid(img, lcc=True):
    if lcc:
        img = img.astype(np.uint8)
        nb_components, output, stats, centroids = cv.connectedComponentsWithStats(img, connectivity=4)
        sizes = stats[:, -1]
        if len(sizes) > 2:
            max_label = 1
            max_size = sizes[1]

            for i in range(2, nb_components):
                if sizes[i] > max_size:
                    max_label = i
                    max_size = sizes[i]

            img2 = np.zeros(output.shape)
            img2[output == max_label] = 255
            img = img2

    if len(img.shape) > 2:
        M = cv.moments(img[:,:,1])
    else:
        M = cv.moments(img)

    if M["m00"] == 0:
        return (img.shape[0] // 2, img.shape[1] // 2)
    
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)

def to_polar(input_img, center):
    input_img = input_img.astype(np.float32)
    max_radius = np.sqrt(((input_img.shape[0]/2.0)**2.0)+((input_img.shape[1]/2.0)**2.0))
    polar_image = cv.linearPolar(input_img, center, max_radius, cv.WARP_FILL_OUTLIERS)
    # polar_image = cv.rotate(polar_image, cv.ROTATE_90_COUNTERCLOCKWISE)
    return polar_image

def to_cart(input_img, center):
    input_img = input_img.astype(np.float32)
    # input_img = cv.rotate(input_img, cv.ROTATE_90_CLOCKWISE)
    max_radius = np.sqrt(((input_img.shape[1]/2.0)**2.0)+((input_img.shape[0]/2.0)**2.0))
    carthesian_image = cv.linearPolar(input_img, center, max_radius, cv.WARP_FILL_OUTLIERS + cv.WARP_INVERSE_MAP)
    carthesian_image = carthesian_image.astype(np.uint8)
    return carthesian_image

def reorient_to_match_training(data_array):
    return np.flipud(np.rot90(data_array, 3))