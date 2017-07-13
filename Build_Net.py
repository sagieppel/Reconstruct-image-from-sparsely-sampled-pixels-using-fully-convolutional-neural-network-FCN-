
import tensorflow as tf
import numpy as np

#################################################################################################################################################################
def CreateConvNet(Sampled_Image,TrainMode=False, CompleteImage=None,NumLayers=16,LayerDepth=128): #Build Net Simple convolutional Mode
   # r, g, b = tf.split(axis=3, num_or_size_splits=3, value=Sampled_Image)
   # Sampled_Image = tf.concat(axis=3, values=[r - 124, g - 117, b - 104,])
#..........................Build First Layer.........................................................................................
    Loss = tf.constant(0, dtype=tf.float32)
    W0 = tf.Variable(tf.truncated_normal([5, 5, 3, 200], mean=0.0, stddev=0.01, dtype=tf.float32), name="W0")
    B0 = tf.Variable(tf.truncated_normal([200], mean=0.0, stddev=0.01, dtype=tf.float32), name="B0")
    Conv0 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(Sampled_Image, W0, [1, 1, 1, 1], padding="SAME"), B0))  # apply covolution add bias and apply relu
    PrevLayerDepth=200 #Depth of previous layer
    MidLayerLossFactor=0#1.2/np.float32(NumLayers)# Loss factor for mid layers prediction (each layer try to predict the final image and have sepearate loss function)
# .............................Build  Middle Layers.................................................................................................................................
    for i in range(NumLayers):
          W1 = tf.Variable(tf.truncated_normal([3, 3, PrevLayerDepth, LayerDepth], mean=0.0, stddev=0.01, dtype=tf.float32), "W"+str(i+1))
          B1 = tf.Variable(tf.truncated_normal([LayerDepth], mean=0.0, stddev=0.01, dtype=tf.float32), "B"+str(i+1))
          Conv1= tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(Conv0, W1, [1, 1, 1, 1], padding="SAME"), B1))
          PrevLayerDepth = LayerDepth
          if TrainMode==True:
              [Rest,RecostructImage]=tf.split(Conv1,[PrevLayerDepth-3,3],axis=3)
              Loss += tf.reduce_mean(tf.abs(RecostructImage-CompleteImage, name="L1_Loss"), name="Loss_"+str(i))* MidLayerLossFactor
          Conv0=Conv1

#...............................Build Final Layer...............................................................................................................................
    WFinal = tf.Variable(tf.truncated_normal([3, 3, PrevLayerDepth, 3], mean=0.0, stddev=0.01, dtype=tf.float32),"WFinal")
    BFinal = tf.Variable(tf.truncated_normal([3], mean=0.0, stddev=0.01, dtype=tf.float32), "BFinal")
    FinalImage = tf.nn.bias_add(tf.nn.conv2d(Conv0, WFinal, [1, 1, 1, 1], padding="SAME"), BFinal)
    if TrainMode == True:
        Loss += tf.reduce_mean(tf.abs(FinalImage-CompleteImage, name="L1_Loss"))
        FinalLoss = tf.reduce_mean(tf.abs(FinalImage - CompleteImage, name="L1_Loss"))
        return FinalImage, Loss, FinalLoss
    else:
        return FinalImage
#########################################Build Net using dilated convultion mode########################################################################################################################
def CreateDilatedNet(Sparse_Sampled_Image,TrainMode=False, CompleteImage=None,NumLayers=18): #Build Net Simple Dilated convolutional Mode
    Loss = tf.constant(0, dtype=tf.float32)
    W0 = tf.Variable(tf.truncated_normal([5, 5, 3, 103], mean=0.0, stddev=0.01, dtype=tf.float32),name = "W0")
    B0 = tf.Variable(tf.truncated_normal([103], mean=0.0, stddev=0.01, dtype=tf.float32), name = "B0")
    Conv0 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(Sparse_Sampled_Image, W0, [1, 1, 1, 1], padding="SAME"), B0))  # apply covolution add bias and apply relu
# ..............................................................................................................................................................
    for i in range(NumLayers):
          W1 = tf.Variable(tf.truncated_normal([3, 3, 103, 103], mean=0.0, stddev=0.01, dtype=tf.float32), "W"+str(i+1))
          B1 = tf.Variable(tf.truncated_normal([103], mean=0.0, stddev=0.01, dtype=tf.float32), "B"+str(i+1))
          Conv1 = tf.nn.bias_add(tf.nn.convolution(Conv0, W1, "SAME", dilation_rate=[i%4+1,i%4+1]), B1)
          if TrainMode==True:
              [Rest,RecostructImage]=tf.split(Conv1,[100,3],axis=3)
              Loss += tf.reduce_mean(tf.abs(RecostructImage-CompleteImage, name="L1_Loss"))*0.05
          Conv0=Conv1
#..............................................................................................................................................................
    WFinal = tf.Variable(tf.truncated_normal([3, 3, 103, 3], mean=0.0, stddev=0.01, dtype=tf.float32),"WFinal")
    BFinal = tf.Variable(tf.truncated_normal([3], mean=0.0, stddev=0.01, dtype=tf.float32), "BFinal")
    FinalImage = tf.nn.bias_add(tf.nn.conv2d(Conv0, WFinal, [1, 1, 1, 1], padding="SAME"), BFinal)
    if TrainMode == True:
        Loss += tf.reduce_mean(tf.abs(FinalImage-CompleteImage, name="L1_Loss"))
        FinalLoss = tf.reduce_mean(tf.abs(FinalImage - CompleteImage, name="L1_Loss"))
        return FinalImage, Loss, FinalLoss
    else:
        return FinalImage

######################################Create net resnet mode#######################################################################
def CreateResNet(Sparse_Sampled_Image, TrainMode=False, CompleteImage=None, NumLayers=8):  # Build Net Resnet Mode
        CompleteImage = tf.cast(CompleteImage, tf.float32)
        # ..........................Build First Layer.........................................................................................
        Loss = tf.constant(0, dtype=tf.float32)
        W0 = tf.Variable(tf.truncated_normal([5, 5, 3, 256], mean=0.0, stddev=0.01, dtype=tf.float32), "W0")
        B0 = tf.Variable(tf.truncated_normal([256], mean=0.0, stddev=0.01, dtype=tf.float32), "B0")
        Conv0 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(Sparse_Sampled_Image, W0, [1, 1, 1, 1], padding="SAME"),
                                          B0))  # apply covolution add bias and apply relu
        # -----------------------Generate Residual neural net main layers---------------------------------------------------------------------------------------
        for i in range(NumLayers):
            W1 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], mean=0.0, stddev=0.01, dtype=tf.float32),
                             name="W" + str(i * 2 + 1))
            B1 = tf.Variable(tf.truncated_normal([256], mean=0.0, stddev=0.01, dtype=tf.float32),
                             name="B" + str(i * 2 + 1))
            Conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(Conv0, W1, [1, 1, 1, 1], padding="SAME"), B1))

            W2 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], mean=0.0, stddev=0.01, dtype=tf.float32),
                             name="W" + str(i * 2 + 2))
            B2 = tf.Variable(tf.truncated_normal([256], mean=0.0, stddev=0.01, dtype=tf.float32),
                             name="B" + str(i * 2 + 2))
            Conv2 = tf.nn.bias_add(tf.nn.conv2d(Conv1, W2, [1, 1, 1, 1], padding="SAME"), B2) + Conv0
            if TrainMode == True:
                [Rest, RecostructImage] = tf.split(Conv2, [253, 3], axis=3)
                Loss += tf.reduce_mean(tf.abs(RecostructImage - CompleteImage, name="L1_Loss"),
                                       name="Loss_" + str(i)) * 0.2
            Conv0 = Conv2
        # ..........................Build First Layer.........................................................................................

        WFinal = tf.Variable(tf.truncated_normal([3, 3, 256, 3], mean=0.0, stddev=0.01, dtype=tf.float32),
                             name="WFinal")
        BFinal = tf.Variable(tf.truncated_normal([3], mean=0.0, stddev=0.01, dtype=tf.float32), name="BFinal")
        FinalImage = tf.nn.bias_add(tf.nn.conv2d(Conv0, WFinal, [1, 1, 1, 1], padding="SAME"), BFinal)
        if TrainMode == True:
            Loss += tf.reduce_mean(tf.abs(FinalImage - CompleteImage, name="L1_Loss"), name="FinalLoss")
            return FinalImage, Loss
        else:
            return FinalImage
    # W3 = tf.Variable(tf.truncated_normal([3, 3, 300, 300], mean=0.0, stddev=0.01, dtype=tf.float32), "W3")
    # B3 = tf.Variable(tf.truncated_normal([400], mean=0.0, stddev=0.01, dtype=tf.float32), "B3")
    # Conv3 = tf.nn.bias_add(tf.nn.conv2d(Conv2, W3, [1, 1, 1, 1], padding="SAME"), B3)+Conv2
    #
    # W4 = tf.Variable(tf.truncated_normal([3, 3, 300, 300], mean=0.0, stddev=0.01, dtype=tf.float32), "W4")
    # B4 = tf.Variable(tf.truncated_normal([400], mean=0.0, stddev=0.01, dtype=tf.float32), "B4")
    # Conv4 = tf.nn.bias_add(tf.nn.conv2d(Conv3, W4, [1, 1, 1, 1], padding="SAME"), B4)+Conv3
    #
    # W5 = tf.Variable(tf.truncated_normal([3, 3, 200, 200], mean=0.0, stddev=0.01, dtype=tf.float32), "W5")
    # B5 = tf.Variable(tf.truncated_normal([400], mean=0.0, stddev=0.01, dtype=tf.float32), "B5")
    # Conv5 = tf.nn.bias_add(tf.nn.convolution(Conv4, W5, "SAME", dilation_rate=[2,2]), B5)
    #
    # W6 = tf.Variable(tf.truncated_normal([3, 3, 300, 300], mean=0.0, stddev=0.01, dtype=tf.float32), "W6")
    # B6 = tf.Variable(tf.truncated_normal([400], mean=0.0, stddev=0.01, dtype=tf.float32), "B6")
    # Conv6 = tf.nn.bias_add(tf.nn.conv2d(Conv5, W6, [1, 1, 1, 1], padding="SAME"), B6)
    #
    # W7 = tf.Variable(tf.truncated_normal([3, 3, 300, 300], mean=0.0, stddev=0.01, dtype=tf.float32), "W7")
    # B7 = tf.Variable(tf.truncated_normal([400], mean=0.0, stddev=0.01, dtype=tf.float32), "B7")
    # Conv7 = tf.nn.bias_add(tf.nn.convolution(Conv6, W7, "SAME", dilation_rate=[4,4]), B7)
    #
    # W8 = tf.Variable(tf.truncated_normal([3, 3, 300, 300], mean=0.0, stddev=0.01, dtype=tf.float32), "W8")
    # B8 = tf.Variable(tf.truncated_normal([300], mean=0.0, stddev=0.01, dtype=tf.float32), "B8")
    # Conv8 = tf.nn.bias_add(tf.nn.conv2d(Conv7, W8, [1, 1, 1, 1], padding="SAME"), B8)
    #
    # W9 = tf.Variable(tf.truncated_normal([3, 3, 300, 300], mean=0.0, stddev=0.01, dtype=tf.float32), "W9")
    # B9 = tf.Variable(tf.truncated_normal([300], mean=0.0, stddev=0.01, dtype=tf.float32), "B9")
    # Conv9 = tf.nn.bias_add(tf.nn.convolution(Conv8, W9, "SAME", dilation_rate=[8,8]), B9)
    #
    # W10 = tf.Variable(tf.truncated_normal([1, 1, 300, 3], mean=0.0, stddev=0.01, dtype=tf.float32), "W10")
    # B10 = tf.Variable(tf.truncated_normal([3], mean=0.0, stddev=0.01, dtype=tf.float32), "B10")
    # ReconstructImage = tf.nn.bias_add(tf.nn.conv2d(Conv9, W10, [1, 1, 1, 1], padding="SAME"), B10)



###########################################################################################################################################################
