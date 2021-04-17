# Semantic-Segmentation-of-LIP-Dataset
This project hosts the code for implementing the FCN algorithm using **PyTorch** for Semantic Segmentation.

### LIP Dataset

Look into Person (LIP) is a new large-scale dataset, focus on semantic understanding of person. 

An overview of the LIP dataset can be found from [the official website](http://sysu-hcp.net/lip/overview.php) . 

Download the LIP dataset from  [Google Drive](https://drive.google.com/drive/folders/0BzvH3bSnp3E9ZW9paE9kdkJtM3M?usp=sharing) or [Baidu Drive](http://pan.baidu.com/s/1nvqmZBN) .

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

### reference

This code refers to  [EMANet](https://github.com/XiaLiPKU/EMANet) or [FCN](https://github.com/Charmve/Semantic-Segmentation-PyTorch) .

