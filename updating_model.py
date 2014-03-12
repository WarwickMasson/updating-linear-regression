'''
This file implements a linear regression model that can
be updated sample by sample.
This avoids the recalculating the coefficients at each step.
'''

import numpy as np

DELTA = 0.000001

class UpdatingModel:
    '''
    UpdatingModel implements a linear regression model
    which can take samples one by one or in a batch.
    '''

    def __init__(self, indim, outdim, delta = DELTA):
        '''
        indim is the number of input dimensions
        outdim is the number of output dimensions
        delta is used to initialize the morr matrix
        '''
        self.morr = np.eye(indim, indim) / delta
        self.beta = np.zeros((indim, outdim)) 
        self.coeff = np.zeros((outdim, indim))
        self.indim = indim
        self.outdim = outdim
        self.shape = (indim, outdim)

    def add_sample(self, inp, out):
        '''
        add_sample adds a single input, output pair to the model.
        input and output should be vectors.
        '''
        inp = to_matrix(inp)
        out = to_matrix(out)
        psi = inp.T.dot(self.morr)
        self.morr -= (self.morr.dot(inp).dot(psi)) / (1 + psi.dot(inp))
        self.beta += inp.dot(out.T)
        self.coeff = self.morr.dot(self.beta).T
        
    def add_samples(self, inputs, outputs):
        '''
        add_samples adds n samples, organized in n rows.
        '''
        for i in range(inputs.shape[0]):
            inp = inputs[i, :]
            out = outputs[i, :]
            self.add_sample(inp, out)

    def predict(self, inp):
        '''
        predict gives the output of the model given an input vector.
        '''
        print to_matrix(inp).shape
        print self.coeff.shape
        return self.coeff.dot(to_matrix(inp))[:, 0]
        

def to_matrix(vector):
    '''
    to_matrix converts a numpy vector into a n x 1 matrix.
    '''
    return np.array([vector]).T
