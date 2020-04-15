"""
@author: jsaavedr
Description: A list of function to create tfrecords from a jfile
"""
import sys
#change the path to .../CC7221/convnet
sys.path.append("/home/jsaavedr/Research/git/public/CC7221/convnet")
import argparse
import utils.configuration as conf
import datasets.data as data

if __name__ == '__main__':                
    
    parser = argparse.ArgumentParser(description = "Create a dataset for training an testing")
    """ pathname must include train.txt and test.txt files """  
    parser.add_argument("-type", type = int, help = "<int> 0: only train, 1: only test, 2: both", required = True )
    parser.add_argument("-config", type = str, help = "<str> configuration file", required = True)
    parser.add_argument("-name", type=str, help=" name of section in the configuration file", required = True)        
    parser.add_argument("-imheight", type = int, help = "<int> size of the image", required = True)
    parser.add_argument("-imwidth", type = int, help = "<int> size of the image", required = False)    
    pargs = parser.parse_args() 
    imh = pargs.imheight
    imw = pargs.imwidth
    if imw is None:
        imw = pargs.imheight
        
    configuration_file = pargs.config
    #assert os.path.exists(configuration_file), "configuration file does not exist {}".format(configuration_file)   
    configuration = conf.ConfigurationFile(configuration_file, pargs.name)               
    #data.create_tfrecords(configuration.get_data_dir(), pargs.type, (imh, imw), number_of_channels = configuration.get_number_of_channels())
    data.create_tfrecords(configuration.get_data_dir(), 
                          pargs.type, 
                          (imh, imw, configuration.get_number_of_channels()))
    print("tfrecords created for " + configuration.get_data_dir())
