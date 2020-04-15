from configparser import SafeConfigParser

class ConfigurationFile:
    """
     An instance of ConfigurationFile contains required parameters to train a 
     convolutional neural network
    """    
    def __init__(self, str_config, modelname):
        config = SafeConfigParser()
        config.read(str_config)
        self.sections = config.sections()                
        if modelname in self.sections:
            try :
                self.modelname = modelname                
                #self.arch = config.get(modelname, "ARCH")
                self.process_fun = 'default'
                if 'PROCESS_FUN' in config[modelname] is not None :
                    self.process_fun = config[modelname]['PROCESS_FUN']                                                    
                self.number_of_classes = config.getint(modelname,"NUM_CLASSES")
                self.number_of_epochs= config.getint(modelname,"NUM_EPOCHS")                                
                self.batch_size = config.getint(modelname, "BATCH_SIZE")
                self.snapshot_steps = config.getint(modelname, "SNAPSHOT_STEPS")
                #test time sets when test is run (in seconds)
                self.test_time = config.getint(modelname, "TEST_TIME")
                self.validation_steps = config.getint(modelname, "VALIDATION_STEPS")
                self.lr = config.getfloat(modelname, "LEARNING_RATE")
                #snapshot folder, where training data will be saved
                self.snapshot_prefix = config.get(modelname, "SNAPSHOT_DIR")
                self.data_dir = config.get(modelname,"DATA_DIR")
                self.channels = config.getint(modelname,"CHANNELS")                                
                assert(self.channels == 1 or self.channels == 3)                
            except Exception:
                raise ValueError("something wrong with configuration file " + str_config)
        else:
            raise ValueError(" {} is not a valid section".format(modelname))
        
    def get_model_name(self):
        return self.modelname
    
    def get_process_fun(self):
        return self.process_fun
       
    def get_number_of_classes(self) :
        return self.number_of_classes
    
    def get_number_of_epochs(self):
        return self.number_of_epochs
    
    def get_batch_size(self):
        return self.batch_size
   
    def get_snapshot_steps(self):
        return self.snapshot_steps
    
    def get_test_time(self):
        return self.test_time
    
    def get_snapshot_dir(self):
        return self.snapshot_prefix
    
    def get_number_of_channels(self):
        return self.channels
    
    def get_data_dir(self):
        return self.data_dir
    
    def get_learning_rate(self):
        return self.lr
    
    def get_validation_steps(self):
        return self.validation_steps      
    
    def is_a_valid_section(self, str_section):
        return str_section in self.sections
    
    def show(self):
        print("NUM_EPOCHS: {}".format(self.get_number_of_epochs()))        
        print("LEARNING_RATE: {}".format(self.get_learning_rate()))                
        print("SNAPSHOT_DIR: {}".format(self.get_snapshot_dir()))
        print("DATA_DIR: {}".format(self.get_data_dir()))
