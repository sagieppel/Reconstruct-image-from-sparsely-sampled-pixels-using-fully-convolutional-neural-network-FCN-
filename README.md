Reconstruct image from sparsely sampled pixels using fully convolutional neural network (FCN)
 
Take image in which only small fraction of the pixels are known and reconstruct (upsample) the full image using fully convolutional neural net (tensorflow implementation).  
 

Training: 
In train.py:
Set folder with  images in: Train_Image_Dir
Set Images size in: Im_Width and Im_Hight
Set Sampling rate (hence fraction of pixels) sampled from each image in: SamplingRate
Run Script the trained net will appear in the log_dir 
 
Using trained net to reconstruct sparsely sample image:
In: RunPrediction.py
 
Assume that you already have trained model in log_dir, if you dont have trained model see training section
Set folder with  images in: Image_Dir
Set Sampling rate (hence fraction of pixels) sampled from each image in: SamplingRate, If the images are already sampled (hence most pixels are zero) set SamplingRate value to 1
Set the directory in which you want the output image to appear to: OUTPUT_Dir
 
The code was run on python 2.7 using tensorflow 1.1
This code is based on Fully convolutional neural nets code: https://github.com/shekkizh/FCN.tensorflow
 published by:Sarath Shekkizhar
 
 
 
 

