liver_config = {
    "mode": "2D",
	"data": "liver",
	"polar_training": True,
	"num_epochs": 50,
	"lr": 3e-05,
	"l2_reg": 1e-05,
	"gpu": 0,
	"num_workers": 8,
	"batch_size": 2,
	"step_size": [25, 42],
	"gamma": 0.1,
	"crop_size": [256, 256],
	"perc_data": 1,
	"train_val_split": 0.9,
	"seed": 1,
	"loss_func": "multiclass_pwce",
	"class_weights": [1, 1],
	"num_classes": 2,
	"augment": ['rot', 'zoom', 'hflip', 'vflip'],
	"verbose_idx": 200,
	"initialization": "",
	"pos_sample_chance": 2,
	"no_standardize": True,
	"epsilon": 1e-06,
	"wmap_weight": 3,
	"weight_score": [1, 1],
	"focal_gamma": 1.5,
	"Training_ROI_Vicinity": 4,
	"savename": "liver_small",
	"use_weightmaps": True,
	"require_one_hot": False,
	"num_out_classes": 2
}

lesion_config = {
    "mode": "2D",
	"data": "lesion",
	"polar_training": True,
	"num_epochs": 55,
	"lr": 3e-05,
	"l2_reg": 1e-04,
	"gpu": 0,
	"num_workers": 8,
	"batch_size": 2,
	"step_size": [45],
	"gamma": 0.2,
	"crop_size": [256, 256],
	"perc_data": 1,
	"train_val_split": 0.9,
	"seed": 1,
	"loss_func": "multiclass_combined",
	"class_weights": [1, 1],
	"num_classes": 2,
	"augment": ['rot', 'zoom', 'hflip', 'vflip'],
	"verbose_idx": 200,
	"initialization": "placeholder/SAVEDATA/Standard_Liver_Networks/vUnet2D_liver_small",
	"pos_sample_chance": 2,
	"no_standardize": True,
	"epsilon": 1e-06,
	"wmap_weight": 3,
	"weight_score": [1, 1],
	"focal_gamma": 1.5,
	"Training_ROI_Vicinity": 4,
	"savename": "liver_small",
	"use_weightmaps": True,
	"require_one_hot": True,
	"num_out_classes": 2
}