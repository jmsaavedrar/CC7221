import sys
#change to ".../CC7221/convnet"
sys.path.append("/home/jsaavedr/Research/git/public/CC7221/convnet")
import tensorflow as tf
from mnist_model import DigitModel
import datasets.data as data
import utils.configuration as conf
import numpy as np
import argparse
import os

if __name__ == '__main__' :    
    parser = argparse.ArgumentParser(description = "Train a simple mnist model")
    parser.add_argument("-config", type = str, help = "<str> configuration file", required = True)
    parser.add_argument("-name", type=str, help=" name of section in the configuration file", required = True)
    pargs = parser.parse_args() 
    configuration_file = pargs.config
    configuration = conf.ConfigurationFile(configuration_file, pargs.name)               
    #parser_tf_record
    #/home/vision/smb-datasets/MNIST-5000/ConvNet2.0/
    tfr_train_file = os.path.join(configuration.get_data_dir(), "train.tfrecords")
    tfr_test_file = os.path.join(configuration.get_data_dir(), "test.tfrecords")
    mean_file = os.path.join(configuration.get_data_dir(), "mean.dat")
    shape_file = os.path.join(configuration.get_data_dir(),"shape.dat")
    #
    input_shape =  np.fromfile(shape_file, dtype=np.int32)
    mean_image = np.fromfile(mean_file, dtype=np.float32)
    mean_image = np.reshape(mean_image, input_shape)
    
    number_of_classes = configuration.get_number_of_classes()
     
    tr_dataset = tf.data.TFRecordDataset(tfr_train_file)
    tr_dataset = tr_dataset.map(lambda x : data.parser_tfrecord(x, input_shape, mean_image, number_of_classes));    
    tr_dataset = tr_dataset.shuffle(5000)        
    tr_dataset = tr_dataset.batch(batch_size = configuration.get_batch_size())    
    #tr_dataset = tr_dataset.repeat()

    
    val_dataset = tf.data.TFRecordDataset(tfr_test_file)
    val_dataset = val_dataset.map(lambda x : data.parser_tfrecord(x, input_shape, mean_image, number_of_classes));    
    val_dataset = val_dataset.batch(batch_size = configuration.get_batch_size())
                
    
    #DigitModel is instantiated
    model = DigitModel()
    #build the model indicating the input shape
    model.build((1, input_shape[0], input_shape[1], input_shape[2]))
    model.summary()
    
    #define the training parameters
    #Here, you can test SGD vs Adam
    model.compile(optimizer=tf.keras.optimizers.Adam(), # 'adam'     
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
        
    history = model.fit(tr_dataset, 
                        epochs = configuration.get_number_of_epochs(),
                        #steps_per_epoch = 100,            
                        validation_data=val_dataset,
                        validation_steps = configuration.get_validation_steps())
        
                        
    #save the model              
    #model.save("model")                        
