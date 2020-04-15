"""
@author: jsaavedr
"""
import numpy as np
import skimage.transform as transf


#to uint8
def toUINT8(image) :
    if image.dtype == np.float64 :
        image = image * 255
    elif image.dtype == np.uint16 :
        image = image >> 8        
    image[image<0]=0
    image[image>255]=255
    image = image.astype(np.uint8, copy=False)
    return image

def process_image(image, imsize):
    """
    imsize = (h,w)
    """ 
    image_out = transf.resize(image, imsize)    
    image_out = toUINT8(image_out)
    return image_out