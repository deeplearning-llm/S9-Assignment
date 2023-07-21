# S9-Assignment

This Assignment Aims at building a Image Classifier Using CNN and Variations of Convolutions such as Depthwise Convolution, Dilated Convolution. We will be using some of the tricks from the Albumentations library for performing data Augmentation. This Assignments however expects the following:
- Total Receptive Field must be more than 44
- One of the layers must use Depthwise Separable Convolution
- One of the layers must use Dilated Convolution
- use GAP
- Achieve 85% accuracy with liberty on the number of epochs which can be used. However the Total Parameters to be less than 200k.

### Datasets
- We will be experimenting with the CIFAR10 Dataset. The images in CIFAR-10 are of size 3x32x32, i.e. 3-channel color images of 32x32 pixels in size. The Images belongs to 10 
  Classes i.e[‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’.]

### Data Pre-Processing
- The following Data Transformations from the Albumentations library are applied 
  1) HorizontalFlip()
  2) ShiftScaleRotate()
  3) CoarseDropout()
  4) Resize()

 Lets have a look at the images for pre-processing.

 ![image](https://github.com/deeplearning-llm/S9-Assignment/assets/135349271/9cf5c8f9-c0ed-40e1-81e1-39bef42f4a98)

 ### CNN Model Architecture.
 We will be bulding the CNN model using the a sepecified architecture using variations of convolutions such as dilated convolution , Depthwise Convolution, GAP Layers. We will not be using any Max Pooling here. 
The following is the architecture for the CNN model. 

Block 1 : Conv1 --> Depthwise-Convolution 2 --> Dilated  Strided Convolution 3
Block 2 : Depthwise-Convolution 4 --> Depthwise-Convolution 5 --> Dilated Strided Convolution 6
Block 3 : Depthwise-Convolution 7 --> Depthwise-Convolution 8 --> Dilated Strided Convolution 9
Block 4 : Depthwise-Convolution 10 --> Depthwise-Convolution 11 --> Dilated Strided Convolution 12 
GAP --> Conv 13

The Architecture contains a total of 198,776 Params.

The CNN is trained for 120 Epochs and the following metrics is observed :

### Model Metrics:
          1) Train Accuracy :  85.98% , Train Loss : 0.4154
          2) Test Accuracy  :  85.24% ,  Test Loss : 0.4631
          
### Learning Curves

![image](https://github.com/deeplearning-llm/S9-Assignment/assets/135349271/21588cc0-fe99-41a2-95a1-ab64b1f1829a)





