import numpy as np
import perceptron

if __name__  == '__main__' :
    w = np.array([-0.06697434, -0.26207466,  2.08373514])
    x = np.array([1, 5.5, 0])
    a = np.dot(w,x)
    print(perceptron.logsig(a))