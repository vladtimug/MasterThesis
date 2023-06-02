import constants
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import cv2 as cv

def set_bounds(image,min_bound,max_bound):
    """
    Clip image to lower bound min_bound, upper bound max_bound.
    """
    return np.clip(image, min_bound, max_bound)

def normalize(image,use_bd=True,zero_center=True,unit_variance=True,supply_mode="orig"):
    """
    Perform standardization/normalization, i.e. zero_centering and Setting
    the data to unit variance.
    Input Arguments are self-explanatory except for:
    supply_mode: Describes the type of LiTS-Data, i.e. whether it has been
                 rescaled/resized or not. See >Basic_Parameter_Values<
    """
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
        image = image - constants.DATASET_MEAN
    if unit_variance:
        image = image / constants.DATASET_STD
    return image

def rotate_2D(to_aug, rng=np.random.RandomState(1)):
    """
    Perform standard 2D-per-slice image rotation.
    Arguments:
    to_aug:     List of files that should be deformed in the same way. Each element
                must be of standard Torch_Tensor shape: (C,W,H,...).
                Deformation is done equally for each channel, but differently for
                each image in a batch if N!=1.
    rng:        Random Number Generator that can be provided for the Gaussian filter means.
    copy_files: If True, copies the input files before transforming. Ensures that the actual
                input data remains untouched. Otherwise, it is directly altered.

    Function only returns data when copy_files==True.
    """
    angle = (rng.rand()*2-1)*10
    for i,aug_file in enumerate(to_aug):
        for ch in range(aug_file.shape[0]):
            #actually perform rotation
            aug_file[ch,:]    = ndi.rotate(aug_file[ch,:].astype(np.float32), angle, reshape=False, order=0, mode="nearest")
    return to_aug, angle

def zoom_2D(to_aug, rng=np.random.RandomState(1)):
    """
    Perform standard 2D per-slice zooming/rescaling.
    Arguments:
    to_aug:     List of files that should be deformed in the same way. Each element
                must be of standard Torch_Tensor shape: (N,C,W,H,...).
                Deformation is done equally for each channel, but differently for
                each image in a batch if N!=1.
    rng:        Random Number Generator that can be provided for the Gaussian filter means.
    copy_files: If True, copies the input files before transforming. Ensures that the actual
                input data remains untouched. Otherwise, it is directly altered.

    Function only returns data when copy_files==True.
    Note: Should also work for 3D, but has not been tested for that.
    """
    # TODO: Figure out how the magnification range limits are computed
    magnif = rng.uniform(0.825,1.175)
    for i,aug_file in enumerate(to_aug):
        for ch in range(aug_file.shape[0]):
            sub_img     = aug_file[ch,:]
            # sub_mask    = aug_file[ch,:]
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
    """
    Perform standard 2D per-slice horizontal_flipping.
    Arguments:
    to_aug:     List of files that should be deformed in the same way. Each element
                must be of standard Torch_Tensor shape: (N,C,W,H,...).
                Deformation is done equally for each channel, but differently for
                each image in a batch if N!=1.
    rng:        Random Number Generator that can be provided for the Gaussian filter means.
    copy_files: If True, copies the input files before transforming. Ensures that the actual
                input data remains untouched. Otherwise, it is directly altered.

    Function only returns data when copy_files==True.
    Note: Should also work for 3D, but has not been tested for that.
    """
    for i,aug_file in enumerate(to_aug):
        for ch in range(aug_file.shape[0]):
            aug_file[ch,:]  = np.fliplr(aug_file[ch,:])

    return to_aug

def vflip_2D(to_aug, rng=np.random.RandomState(1)):
    """
    Perform standard 2D per-slice vertical flipping.
    Arguments:
    to_aug:     List of files that should be deformed in the same way. Each element
                must be of standard Torch_Tensor shape: (N,C,W,H,...).
                Deformation is done equally for each channel, but differently for
                each image in a batch if N!=1.
    rng:        Random Number Generator that can be provided for the Gaussian filter means.
    copy_files: If True, copies the input files before transforming. Ensures that the actual
                input data remains untouched. Otherwise, it is directly altered.

    Function only returns data when copy_files==True.
    Note: Should also work for 3D, but has not been tested for that.
    """
    for i,aug_file in enumerate(to_aug):
        for ch in range(aug_file.shape[0]):
            aug_file[ch,:]  = np.flipud(aug_file[ch,:])

    return to_aug

def augment_2D(to_aug, mode_dict=["rot","zoom"], copy_files=False, return_files=False, seed=1, is_mask=[0,1,0]):
    """
    Combine all augmentation methods to perform data augmentation (in 2D). Selection is done randomly.
    Arguments:
    to_aug:     List of files that should be deformed in the same way. Each element is a list with
                Arrays of standard Torch_Tensor shape: (C,W,H,...).
                Augmentation is done equally for each channel, but differently for
                each image in a batch if N!=1.
    mode_dict:  List of augmentation methods that should be used.
    rng:        Random Number Generator that can be provided for the Gaussian filter means.
    copy_files: If True, copies the input files before transforming. Ensures that the actual
                input data remains untouched. Otherwise, it is directly altered.

    Function only returns data when copy_files==True.
    """
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
    """
    Function to crop from input images.
    Takes as input a list of same-shaped 3D/4D-arrays with Ch,W,H(,D). If an index-file
    is supplied, crops will only be taken in and around clusters in the index file. If the index-file
    contains no clusters, then a random crop will be taken.

    Arguments:
    batch_to_crop:      list of batches that need to be cropped. Note that cropping is performed independently for
                        each image of a batch.
    idx_batch:          Batch of same size as input batches. Contains either clusters (i.e. ones) from which a
                        cluster-center will be sampled or None. In this case, the center will be randomly selected.
                        If not None, prov_coords must be None. The idx_image should ahve shape (1,W,H).
    crop_size:          Size of the crops to take -> len(crop_size) = input_batch.ndim-1, i.e. ignore batchdimension.
    """
    rng = np.random.RandomState(seed)

    sup = list(1-np.array(crop_size)%2)
    bl_len = len(batch_to_crop)
    batch_list_to_return = []

    ### Provide idx-list
    batch_list_to_return_temp = [[] for i in range(len(batch_to_crop))]

    if idx_batch is not None:
        all_crop_idxs = np.where(idx_batch[0,:]==1) if np.sum(idx_batch[0,:])!=0 else [[]]
    else:
        all_crop_idxs = [[]]

    if len(all_crop_idxs[0]) > 0:
        if idx_batch is not None:
            crop_idx = [np.clip(rng.choice(ax),crop_size[i]//2-1,batch_to_crop[0][:].shape[i+1]-crop_size[i]//2-1) for i,ax in enumerate(all_crop_idxs)]
    else:
        crop_idx = [rng.randint(crop_size[i]//2-1,np.array(batch_to_crop[0].shape[i+1])-crop_size[i]//2-1) for i in range(batch_to_crop[0].ndim-1)]
    
    crop_coordinates = [(center - crop_size[i] // 2 + mv, center + crop_size[i] // 2 + 1) for i, (center, mv) in enumerate(zip(list(crop_idx),sup))]
    
    for i in range(bl_len):
        batch_list_to_return.append(batch_to_crop[i][:, crop_coordinates[0][0]: crop_coordinates[0][1], crop_coordinates[1][0]: crop_coordinates[1][1]])

    return tuple(batch_list_to_return)

def numpy_generate_onehot_matrix(matrix_mask, ndim):
    """
    Function to convert a mask array of shape W,H(,D) with values
    in 0...C-1 to an array of shape C,W,H(,D). Works with numpy arrays.

    Arguments:
        matrix_mask:    Mask to convert.
        ndim:           Number of additional one-hot dimensions.
    """
    onehot_matrix = np.eye(ndim)[matrix_mask.reshape(-1).astype('int')].astype('int')
    data_shape    = list(matrix_mask.shape)
    data_shape[0] = ndim
    onehot_matrix = np.fliplr(np.flipud(onehot_matrix).T).reshape(*data_shape)
    return onehot_matrix

def centroid(img, lcc=False):
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
