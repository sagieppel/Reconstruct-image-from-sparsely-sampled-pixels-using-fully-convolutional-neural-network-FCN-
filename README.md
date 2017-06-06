# Reconstruct image from sparsely sampled pixels using fully convolutional neural network (FCN)
 
Take image in which only small fraction of the pixels are known and reconstruct (upsample) the full image using fully convolutional neural net (tensorflow implementation).  
![](/Image1.png)  ![](/Scheme.png)
## Instructions
### Training: 
In train.py:
1) Set folder with  images in: Train_Image_Dir
2) Set Images size in: Im_Width and Im_Hight
3) Set Sampling rate (hence fraction of pixels) sampled from each image in: SamplingRate
4) Run Script. The trained net will appear in the log_dir 
 
### Using trained net to reconstruct sparsely sample image:
In: RunPrediction.py
 
Assume that you already have trained model in log_dir, if you dont have trained model see training section
1) Set folder with  images in: Image_Dir
2) Set Sampling rate (hence fraction of pixels) sampled from each image in: SamplingRate, If the images are already sampled set SamplingRate value to 1
3) Set the directory in which you want the output image to appear to: OUTPUT_Dir
 
The code was run on python 2.7 using tensorflow 1.1
This code is based on Fully convolutional neural nets code: https://github.com/shekkizh/FCN.tensorflow
 published by:Sarath Shekkizhar
 
 
 
 

