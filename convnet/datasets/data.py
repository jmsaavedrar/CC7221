"""
@author: jsaavedra

Module containing data operations
"""
import os
import sys
import numpy as np
import random
import utils.imgproc as imgproc
import tensorflow as tf
import skimage.io as io
import skimage.color as color


#%% int64 should be used for integer numeric values
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#%% byte should be used for string  | char data
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
#%% float should be used for floating point data
def _float_feature(value):    
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def read_image(filename, number_of_channels):
    """ read_image using skimage
        The output is a 3-dim image [H, W, C]
    """    
    if number_of_channels  == 1 :            
        image = io.imread(filename, as_gray = True)
        image = imgproc.toUINT8(image)
        assert(len(image.shape) == 2)
        image = np.expand_dims(image, axis = 2) #H,W,C                    
        assert(len(image.shape) == 3 and image.shape[2] == 1)
    elif number_of_channels == 3 :
        image = io.imread(filename)
        if(len(image.shape) == 2) :
            image = color.gray2rgb(image)
        image = imgproc.toUINT8(image)        
        assert(len(image.shape) == 3)
    else:
        raise ValueError("number_of_channels must be 1 or 3")
    if not os.path.exists(filename):
        raise ValueError(filename + " does not exist!")
    return image

def validate_labels(labels) :
    """It checks if labels are in the correct format [int]
       labels need to be integers, from 0 to NCLASSES -1 
    """  
    new_labels = [int(label) for label in labels]
    label_set = set(new_labels)
    #checking the completness of the label set
    if (len(label_set) == max(label_set) + 1) and (min(label_set) == 0):
        return new_labels
    else:            
        raise ValueError("Some codes are missed in label set! {}".format(label_set))
    
def read_data_from_file(str_path, dataset = "train" , shuf = True):    
    """read data from text files
    and apply shuffle by default 
    """            
    datafile = os.path.join(str_path, dataset + ".txt")    
    assert os.path.exists(datafile)        
    # reading data from files, line by line
    with open(datafile) as file :        
        lines = [line.rstrip() for line in file]     
        if shuf:
            random.shuffle(lines)
        _lines = [tuple(line.rstrip().split('\t'))  for line in lines ] 
        filenames, labels = zip(*_lines)
        labels = validate_labels(labels)    
    return filenames, labels

def create_tfrecords_from_file(filenames, labels, target_shape, tfr_filename, process_function = imgproc.process_image):
    h = target_shape[0]
    w = target_shape[1]
    number_of_channels = target_shape[2]
    #create tf-records
    writer = tf.io.TFRecordWriter(tfr_filename)
    #filenames and lables should  have the same size    
    assert len(filenames) == len(labels)
    mean_image = np.zeros([h,w,number_of_channels], dtype=np.float32)    
    for i in range(len(filenames)):        
        if i % 500 == 0 or (i + 1) == len(filenames):
            print("---{}".format(i))           
        image = read_image(filenames[i], number_of_channels)
        image = process_function(image, (h, w))
        #print(image)
        #cv2.imshow("image", image)
        #print(" {} {} ".format(image.shape, labels[i]))        
        #cv2.waitKey()        
        #create a feature                
        feature = {'image': _bytes_feature(tf.compat.as_bytes(image.tostring())),
                   'label': _int64_feature(labels[i])}
        
        #create an example protocol buffer
        example = tf.train.Example(features = tf.train.Features(feature=feature))        
        #serialize to string and write on the file
        writer.write(example.SerializeToString())
        mean_image = mean_image + image / len(filenames)            
    writer.close()
    sys.stdout.flush()
    return mean_image

def create_tfrecords(data_dir, type_id, image_shape, processFun = imgproc.process_image):    
    """ 
    data_dir
    type_id:0 = only train 
            1 = only test    
            2 = both
    im_shape: [H,W,C] of the input          
    processFun: processing function which depends on the problem we are dealing with
    """
    image_shape = np.asarray(image_shape)
    #saving metadata   
    #------------- creating train data
    if ( type_id + 1 ) & 1 : # train   ( 0 + 1 ) & 1  == 1 
        filenames, labels = read_data_from_file(data_dir, dataset="train", shuf = True)    
        tfr_filename = os.path.join(data_dir, "train.tfrecords")
        training_mean = create_tfrecords_from_file(filenames, labels, image_shape, tfr_filename, processFun)
        print("train_record saved at {}.".format(tfr_filename))
        #saving training mean
        mean_file = os.path.join(data_dir, "mean.dat")
        print("mean_file {}".format(training_mean.shape))
        training_mean.astype(np.float32).tofile(mean_file)
        print("mean_file saved at {}.".format(mean_file))  
    #-------------- creating test data    
    if ( type_id + 1 ) & 2 : # test ( 1 + 1 ) & 2  == 2
        filenames, labels = read_data_from_file(data_dir, dataset="test", shuf = True)  
        tfr_filename = os.path.join(data_dir, "test.tfrecords")
        create_tfrecords_from_file(filenames, labels, image_shape, tfr_filename, processFun)
        print("test_record saved at {}.".format(tfr_filename))    
            
    #saving shape file    
    shape_file = os.path.join(data_dir, "shape.dat")
    image_shape.astype(np.int32).tofile(shape_file)
    print("shape_file saved at {}.".format(shape_file))
    
#parser tf_record to be used for dataset mapping
def parser_tfrecord(serialized_input, input_shape, mean_image, number_of_classes):
        features = tf.io.parse_example([serialized_input],
                                features={
                                        'image': tf.io.FixedLenFeature([], tf.string),
                                        'label': tf.io.FixedLenFeature([], tf.int64)
                                        })
        #image
        image = tf.io.decode_raw(features['image'], tf.uint8)
        image = tf.reshape(image, input_shape)        
        image = tf.cast(image, tf.float32) - mean_image
        #label
        label = tf.cast(features['label'], tf.int32)
        label = tf.one_hot(label, depth = number_of_classes)
        label = tf.reshape(label, [number_of_classes])
        
        return image, label          