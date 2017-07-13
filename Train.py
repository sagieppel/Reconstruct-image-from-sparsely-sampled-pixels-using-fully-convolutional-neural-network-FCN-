#Script for training network for recosttucting image from sparsly sample pixels
#Hence take image in which only small fraction of the pixels are known and reconstruct the full image
#The unknown pixels are marked as 0

#Instructions for training
#Set folder with  images in: Train_Image_Dir
#Set Images size in: Im_Width and Im_Hight
#Set Sampling rate (hence fraction of pixels) sampled from each image in: SamplingRate
#Run training: the train model will appear in the log file
#To use the trained network for reconstructing image from sparsely sampled pixels see: RunPrediction.py

from __future__ import print_function
import tensorflow as tf
import numpy as np
import Build_Net as BuildNet #Were the net is built
import ImageReader as ImageReader # loader for images
import os
import scipy.misc as misc
import random

Train_Image_Dir="/media/sagi/1TB/Data_zoo/COCO/test2015/"# Directory with image to train
SamplingRate=0.1 # Fraction of pixels to be sampled from image for training
MaxImageSize=250000

Batch_Size=1 # For training (number of images trained per iteration)
logs_dir="logs/" # Were the trained model and all output will be put
learning_rate=1e-5#Learning rate for Adam Optimizer
Vgg_Model_Dir="Model_zoo/" #Directory of the pretrained VGG model if model not there it will be automatically download
TrainLossTxtFile=logs_dir+"TrainLoss.txt"#

#######If you use batch of more then one you must set standart size for image#####################################3
Resize=False
ImageWidth=400
ImageHeight=400
if Batch_Size>1: Resize=True # if using batch size of more then one all images will be resized at training
######################################################################################3

MAX_ITERATION = int(300000) #Maximal training iteration

if not os.path.exists(logs_dir): os.makedirs(logs_dir)

###########################################Solver for the net training################################################################################################################
def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    return optimizer.apply_gradients(grads)

################################################################################################################################################################################
def main(argv=None):

    Sparse_Sampled_Image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_Sparse_image") #Input image sparsly sampled image
    Full_Image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="Full_image")  # Full image all pixels filled

    [ReconstructImage,Loss,FinalLoss] = BuildNet.CreateDilatedNet(Sparse_Sampled_Image,TrainMode=True, CompleteImage=Full_Image)# Here the graph(net) is builded


    # tf.summary.scalar("L1_Loss", loss)

    trainable_var = tf.trainable_variables()
    train_op = train(Loss, trainable_var)

    #print("Setting up summary op...")
    #summary_op = tf.summary.merge_all()

    print("Reading images list")
    TrainImages=[]   #Train Image List

    TrainImages += [each for each in os.listdir(Train_Image_Dir) if each.endswith('.PNG') or each.endswith('.JPG') or each.endswith('.TIF') or each.endswith( '.GIF') or each.endswith('.png') or each.endswith('.jpg') or each.endswith('.tif') or each.endswith('.gif')]  # Get list of training images


    print('Number of  Train images='+str(len(TrainImages)))

#-------------------------Training Region-----------------------------------------------------------------------------------------------------------------------------

    sess = tf.Session() #Start Tensorflow session

    print("Setting up Saver...")
    saver = tf.train.Saver()
   # summary_writer = tf.summary.FileWriter(logs_dir, sess.graph)
    sess.run(tf.global_variables_initializer())
    #sess.run(tf.initialize_all_variables())
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path: # if trained model exist restore it
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
#---------------------------Start Training: Create loss files----------------------------------------------------------------------------------------------------------
    Nimg=0
    f = open(TrainLossTxtFile, "w")# Create text file for writing the loss trough out the training
    f.write("Iteration\tTrain_Loss\t Learning Rate="+str(learning_rate))
    f.close()
#-----------------------------------------------------------------------------------------------------------------
    Epoch=0
#..............Start Training loop: Main Training....................................................................
    for itr in range(1,MAX_ITERATION+1):
        if Nimg>=len(TrainImages)-1: # End of an epoch
           Nimg=0
           random.shuffle(TrainImages)  #Suffle images every epoch
           Epoch += 1
           print("Epoch "+str(Epoch)+" Completed")
#.....................Load images for training
        batch_size=np.min([Batch_Size,len(TrainImages)-Nimg])
  #---------------------if resize image before training not recommanded
        if Resize==True: # If resize image before training
           FullImages = np.zeros([batch_size,ImageHeight,ImageWidth,3], dtype=np.int)
           SparseSampledImages = np.zeros([batch_size,ImageHeight,ImageWidth,3], dtype=np.int)
           for fi in range(batch_size):
               FullImages[fi],SparseSampledImages[fi]=ImageReader.LoadImages(Train_Image_Dir +TrainImages[Nimg],SamplingRate,Resize=True,Im_Hight=ImageHeight,Im_Width=ImageWidth)
               Nimg+=1
  #-----------------------------------------------------------------------------------------------------------------------------------------------
        else:
            FullImages, SparseSampledImages = ImageReader.LoadImages(Train_Image_Dir + TrainImages[Nimg],SamplingRate,MaxSize=MaxImageSize)
            Nimg += 1
        #.......................Run one batch of training...............................................................................
        feed_dict = {Sparse_Sampled_Image: SparseSampledImages, Full_Image: FullImages}# Run one cycle of traning
        sess.run(train_op, feed_dict=feed_dict)
#......................Write training set loss..........................................................................
        if itr % 10==0:
            feed_dict = {Sparse_Sampled_Image: SparseSampledImages, Full_Image: FullImages}
            train_loss,FinalLayerTrainLoss= sess.run([Loss,FinalLoss], feed_dict=feed_dict)
            print("Step: "+str(itr)+") Train_loss_All_Layers="+str(train_loss)+", Final Layer Loss="+str(FinalLayerTrainLoss))
          #  summary_writer.add_summary(summary_str, itr)
            with open(TrainLossTxtFile, "a") as f:#Write training loss for file
                 f.write("\n"+str(itr)+"\t"+str(train_loss))
                 f.close()
#....................Save Trained net (ones every 1000 training cycles...............................................
        if itr%500==0 and itr>0:
            print("Saving Model")
            saver.save(sess, logs_dir + "model.ckpt", itr)# save trained model
print("Finished Running")
if __name__ == "__main__":
    tf.app.run()
