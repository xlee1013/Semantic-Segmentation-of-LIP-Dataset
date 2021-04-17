# Semantic-Segmentation-of-LIP-Dataset
This project hosts the code for implementing the FCN algorithm for Semantic Segmentation.

### Dataset directory

```
LIP_DATA
----datalist
--------test_id.txt
--------train_id.txt
--------val_id.txt
----Testing_images
--------10,000 images
----TrainVal_images
--------train_images
------------30,462 images
--------val_images
------------10,000 images
----Trainval_parsing_annotations
--------train_segmentations
------------30,462 images
--------val_segmentations
------------10,000 images
```

### Dependencies

```
pytorch >= 1.5.0
```

### Settings

The **settings.py** file defines all the parameters required for data processing, model training, verification and testing phases. 

### Train

```
python train.py
```

###  Verification

```
python eval.py
```

### Test

```
python test.py
```