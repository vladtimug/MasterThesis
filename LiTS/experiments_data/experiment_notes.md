# Experiments Notes

## This document is intended to hold information related to the experimental data and the experimental setup used to acquire the data for the task of liver tumor semantic segmentation.

<br>
--------------------------------------------------------------------------
</br>

### <b>1. Experiments Set 1 (Custom UNet)</b>
<font size="3">
    This set of experiments was mainly used in order to figure out what is the most efficient crop size to use during training in order to attain a reasonable tradeoff betweeen model performance and training time.<br><br>
    <b>
        * The models in this set of experiments were trained on carthesian coordinates.<br>
        * The tumor segmentation model was initialized with the weights of the trained liver segmentation model.<br><br>
    </b>
    The following criteria is critical for discriminating between experiments and holds true for both the liver and lesion experiments:
    
    1. experiment_1 - UNet model trained on crop sizes of 256x256 pixels
    2. experiment_2 - UNet model trained on crop sizes of 128x128 pixels
    3. experiment_3 - UNet model trained on uncropped samples of default size 512x512
</font>

<br>
--------------------------------------------------------------------------
</br>

### <b>2. Experiments Set 2 (Custom UNet)</b>
<font size="3">
    This set of experiments was mainly used in order to figure out if training on polar coordinates for the task of liver/tumor segmentation yields better results than training on carthesian coordinates.<br><br>
    <b>
        * The models in these experiments have been trained on crop sizes of 256x256 pixels<br>
        * The tumor segmentation model was initialized with the weights of the trained liver segmentation model.<br><br>
    </b>
    The following criteria is critical for discriminating between experiments:

    1. experiment_4 - UNet model trained on polar transformations of the training samples for liver segmentation. Lesion segmentation model  was initialized with the weights of the liver segmentation model and trained on the same polar transformations applied to the input samples.
    2. experiment_5 - UNet model trained on carthesian coordinates of the input samples for liver segmentation (model from experiment_1). Lesion segmentation model was initialized with the weights of the liver segmentation model and trained on the polar transformations applied to the input samples.
</font>

<br>
--------------------------------------------------------------------------
</br>

### <b>3. Experiments Set 3 (Classic UNet)</b>
<font size="3">
    This set of experiments is intended to compare the results between training a classic UNet model for the task of liver tumor segmentation on the data represented in carthesian coordinates vs training the same model for the same task on the same data in polar coordinates.
    <b>
        * The models in these experiments have been trained on crop sizes of 256x256 pixels<br>
        * The tumor segmentation model was initialized with pytorch default values.
        * The center for the polar transformation was computed by <br><br>
    </b>
    
    1. experiment_6 - Classic UNet model trained on data represented in polar coordinates.
    2. experiment_7 - Classic UNet model trained on data represented in carthesian coordinates.
</font>

<br>
--------------------------------------------------------------------------
</br>

### <b>4. Experiments Set 4 (UNet++)</b>
<font size="3">
    This set of experiments is intended to compare the results between training a custom UNet model for the task of liver tumor segmentation on the data represented in carthesian coordinates vs training the same model for the same task on the same data in polar coordinates.
    <b>
        * The models in these experiments have been trained on crop sizes of 256x256 pixels<br>
        * The tumor segmentation model was initialized with pytorch default values.<br><br>
    </b>
    
    1. experiment_8 - UNet++ model trained on data represented in carthesian coordinates.
    2. experiment_9 - UNet++ model trained on data represented in polar coordinates.
</font>

<br>
--------------------------------------------------------------------------
</br>

### <b>5. Experiments Set 5 (Custom UNet)</b>
<font size="3">
    This set of experiments is intended to compare the results between training a custom UNet model for the task of liver tumor segmentation on the data represented in carthesian coordinates vs training the same model for the same task on the same data in polar coordinates.
    <b>
        * The models in these experiments have been trained on crop sizes of 256x256 pixels<br>
        * The tumor segmentation model was initialized with pytorch default values.<br><br>
    </b>
    
    1. experiment_10 - Custom UNet model trained on data represented in polar coordinates.
    2. experiment_11 - Custom UNet model trained on data represented in carthesian coordinates.
</font>

### <b>5. Experiments Set 5 (DeepLabV3+)</b>
<font size="3">
    This set of experiments is intended to compare the results between training a DeepLabV3+ model for the task of liver tumor segmentation on the data represented in carthesian coordinates vs training the same model for the same task on the same data in polar coordinates.
    <b>
        * The models in these experiments have been trained on crop sizes of 256x256 pixels<br>
        * The tumor segmentation model was initialized with pytorch default values.<br><br>
    </b>
    
    1. experiment_12 - DeepLabV3+ model trained on data represented in carthesian coordinates.
    2. experiment_13 - DeepLabV3+ model trained on data represented in polar coordinates.
</font>