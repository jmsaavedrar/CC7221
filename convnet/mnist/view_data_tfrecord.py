import tensorflow as tf
from tensorflow.keras import datasets
from mnist_model import DigitModel
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__' :
    
    #The datas is loaded, you can also use tf-records 
    
    def parser_tfrecord(serialized_input, image_shape, mean_image, number_of_classes):
        features = tf.io.parse_example([serialized_input],
                                features={
                                        'data/image': tf.io.FixedLenFeature([], tf.string),
                                        'data/label': tf.io.FixedLenFeature([], tf.int64)
                                        })
        image = tf.io.decode_raw(features['data/image'], tf.uint8)
        image = tf.reshape(image, image_shape)        
        #image = tf.cast(image, tf.float32)/255.0
        label = tf.cast(features['data/label'], tf.int32)
        #label = tf.one_hot(label, number_of_classes)
        return image, label
        
    #preparing dataset, tf-records can be used too
    #to read tf-records use tf.data.TFRecordDataset(tf-record-file)
    
    tfr_train_file = "/home/vision/smb-datasets/MNIST-5000/ConvNet2.0/train.tfrecords"
    tfr_test_file = "/home/vision/smb-datasets/MNIST-5000/ConvNet2.0/test.tfrecords"
    mean_file = "/home/vision/smb-datasets/MNIST-5000/ConvNet2.0/mean.dat"
    shape_file = "/home/vision/smb-datasets/MNIST-5000/ConvNet2.0/shape.dat"
    input_shape =  np.fromfile(shape_file, dtype=np.int32)
    mean_image = np.fromfile(mean_file, dtype=np.float32)
    mean_image = np.reshape(mean_image, input_shape)
    print(input_shape)
    print(mean_image.shape)
    number_of_classes = 10
    tr_dataset = tf.data.TFRecordDataset(tfr_train_file)    
    tr_dataset = tr_dataset.map(lambda x : parser_tfrecord(x, input_shape, mean_image, number_of_classes));
    tr_dataset = tr_dataset.shuffle(60000)    
    tr_dataset = tr_dataset.batch(batch_size = 10)    

    fig,xs = plt.subplots(2,5)    
    for batch in tr_dataset :
        images = batch[0]
        labels = batch[1]
        n = images.shape[0]
        for i in range(n) :
            row = int(i / 5)
            col = int(i % 5)
            print(images.shape)            
            xs[row, col].imshow(images[i,:,:,0], cmap = 'gray')
            xs[row, col].set_title(str(labels[i].numpy()))
            xs[row, col].set_axis_off()        
        plt.pause(0.001)
    
