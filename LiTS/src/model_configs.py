liver_config = {
	"model": "custom_unet",	# Available options: classic_unet, custom_unet, unet_plus_plus
	"structure": [3, 3, 3, 3, 3],
	"filter_start": 20,
	"channels": 1,
	"init_type": "he_n",
	"use_batchnorm": True,
	"dropout": 0,
	"use_pypool": False,
	"use_auxiliary_inputs": False,
	"use_conv_pool": True,
	"backbone": "base",
	"block_type": "base",   # Available options: base, res, resX
	"up_conv_type": [4, 2, 1],
	"se_reduction": 0,
	"dilation": [1, 1, 1, 1, 1],
	"dilation_up": [1, 1, 1, 1, 1],
	"structure_up": [3, 3, 3, 3, 3],
	"filter_start_up": 20,
	"filter_sizes": [20, 40, 80, 160, 320, 640],
	"filter_sizes_up": [20, 40, 80, 160, 320, 640],
	"model_name": "vUnet2D"
}

lesion_config = {
	"model": "custom_unet",	# Available options: classic_unet, custom_unet, unet_plus_plus
	"structure": [3, 3, 3, 3, 3],
	"filter_start": 20,
	"channels": 1,
	"init_type": "he_n",
	"use_batchnorm": True,
	"dropout": 0,
	"use_pypool": False,
	"use_auxiliary_inputs": False,
	"use_conv_pool": True,
	"backbone": "base",
	"block_type": "base",
	"up_conv_type": [4, 2, 1],
	"se_reduction": 0,
	"dilation": [1, 1, 1, 1, 1],
	"dilation_up": [1, 1, 1, 1, 1],
	"structure_up": [3, 3, 3, 3, 3],
	"filter_start_up": 20,
	"filter_sizes": [20, 40, 80, 160, 320, 640],
	"filter_sizes_up": [20, 40, 80, 160, 320, 640],
	"model_name": "vUnet2D",
}
