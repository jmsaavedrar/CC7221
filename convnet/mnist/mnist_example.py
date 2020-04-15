import tensorflow as tf
from tensorflow.keras import datasets
from mnist_model import DigitModel


if __name__ == '__main__' :
    
    #The datas is loaded, you can also use tf-records 
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()    
    train_images = tf.cast(train_images, dtype = tf.float32)
    test_images = tf.cast(test_images, dtype = tf.float32)    
    # Normalize pixel values to be between 0 and 1
    # 
    train_images = train_images / 255.0; 
    test_images = test_images / 255.0;
    
    #reshape
    train_images = tf.reshape(train_images, shape = (-1,28,28,1)) # B, H, W, C 
    test_images = tf.reshape(test_images, shape = (-1,28,28,1))
    train_labels = tf.one_hot(train_labels, depth = 10)
    test_labels = tf.one_hot(test_labels, depth = 10) # e.g. 0 = 1 0 0 0 0 0 0 0 0 0
    
    
    #preparing dataset, tf-records can be used too
    #to read tf-records use tf.data.TFRecordDataset(tf-record-file)
    tr_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    tr_dataset = tr_dataset.shuffle(60000)
    tr_dataset = tr_dataset.batch(batch_size = 64)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    val_dataset = val_dataset.shuffle(10000)
    val_dataset = val_dataset.batch(batch_size = 64)           
    
    #DigitModel is instantiated
    model = DigitModel()
    #build the model indicating the input shape
    model.build((1,28,28,1))
    model.summary()
    
    #define the training parameters
    model.compile(optimizer=tf.keras.optimizers.Adam(), # 'adam'     
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
 
     
     
    history = model.fit(tr_dataset, 
                        epochs = 2,
                        validation_data=val_dataset)
                         
    #save the model              
    #model.save("model")                        
