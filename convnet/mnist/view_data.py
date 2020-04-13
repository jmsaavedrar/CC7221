# author: jsaavedr
# Viewing the data
from tensorflow.keras import datasets
import tensorflow as tf
import matplotlib.pyplot as plt


if __name__ == '__main__' :
    
    #The datas is loaded, you can also use tf-records 
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    dataset =  tf.data.Dataset.from_tensor_slices((test_images, test_labels))    
    dataset = dataset.batch(10)
    
    fig,xs = plt.subplots(2,5)    
    for batch in dataset :
        images = batch[0]
        labels = batch[1]
        n = images.shape[0]
        for i in range(n) :
            row = int(i / 5)
            col = int(i % 5)
            print(images[i,:,:].shape)
            xs[row, col].imshow(images[i,:,:], cmap = 'gray')
            xs[row, col].set_title(str(labels[i].numpy()))
            xs[row, col].set_axis_off()        
        plt.pause(0.001)            
        
    