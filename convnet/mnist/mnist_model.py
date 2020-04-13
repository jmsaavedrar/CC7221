# author: jsaavedr
# April, 2020 
# This is py model for mnist 

import tensorflow as tf

class DigitModel(tf.keras.Model):
    def __init__(self):
        super(DigitModel, self).__init__()
        #define layers which require parameters to be learned
        self.conv_1 = tf.keras.layers.Conv2D(32, (3,3), padding = 'valid',  kernel_initializer='he_normal')
        self.max_pool = tf.keras.layers.MaxPool2D((3,3), strides = 2, padding = 'valid')
        self.relu = tf.keras.layers.ReLU();
        self.bn_conv_1 = tf.keras.layers.BatchNormalization()                
        self.conv_2 = tf.keras.layers.Conv2D(64, (3,3), padding = 'valid',  kernel_initializer='he_normal')
        self.bn_conv_2 = tf.keras.layers.BatchNormalization()        
        self.conv_3 = tf.keras.layers.Conv2D(128, (3,3), padding = 'valid', kernel_initializer='he_normal')
        self.bn_conv_3 = tf.keras.layers.BatchNormalization()
        self.fc1 = tf.keras.layers.Dense(256, kernel_initializer='he_normal')
        self.bn_fc_1 = tf.keras.layers.BatchNormalization()
        self.fc2 = tf.keras.layers.Dense(10)
        
    # here the architecture is defined
    def call(self, inputs):
        #first block
        _conv1 = self.conv_1(inputs)
        _conv1 = self.bn_conv_1(_conv1)
        _conv1 = self.relu(_conv1)
        _conv1 = self.max_pool(_conv1)
        #second block
        _conv2 = self.conv_2(_conv1)
        _conv2 = self.bn_conv_2(_conv2)
        _conv2 = self.relu(_conv2)
        _conv2 = self.max_pool(_conv2)
        #third block
        _conv3 = self.conv_3(_conv2)
        _conv3 = self.bn_conv_3(_conv3)
        _conv3 = self.relu(_conv3)
        #last block        
        _conv3_flatten = tf.keras.layers.Flatten()(_conv3)
        _fc1 = self.fc1(_conv3_flatten)
        _fc1 = self.bn_fc_1(_fc1)
        _fc1 = self.relu(_fc1)
        
        output = self.fc2(_fc1)
        return output
