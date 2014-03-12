'''
This file implements tests for the UpdatingModel class.
'''

import numpy as np
from unittest import TestCase
from updating_model import to_matrix, UpdatingModel
import random

ITERATIONS = 100
DIM_SIZE = 20
SAMPLES = 100
DIFF = 0.0001

def random_size():
    '''
    Returns a random in or out dimension size.
    '''
    return random.randint(1, DIM_SIZE)

class TestModel(TestCase):
    '''
    A TestCase class to test the UpdatingModel methods.
    '''

    def test_init(self):
        '''
        Tests the initialization of the model.
        '''
        for _ in range(ITERATIONS):
            indim = random_size()
            outdim = random_size()
            model = UpdatingModel(indim, outdim)
            self.assertTrue(model.coeff.shape == (outdim, indim))
            self.assertTrue(model.morr.shape == (indim, indim))
            self.assertTrue(model.beta.shape == (indim, outdim))

    def test_add_sample(self):
        '''
        Tests the add_sample method.
        '''
        for _ in range(ITERATIONS):
            indim = random_size()
            outdim = random_size()
            coeff = np.random.randn(indim, outdim)
            model = UpdatingModel(indim, outdim)
            for _ in range(SAMPLES):
                inp = np.random.randn(indim)
                out = coeff.T.dot(to_matrix(inp))[:, 0]
                model.add_sample(inp, out)
            diff = sum(sum(abs(model.coeff.T - coeff)))
            self.assertTrue(diff < DIFF)

    def test_add_samples(self):
        '''
        Tests the add_samples method.
        '''
        for _ in range(ITERATIONS):
            indim = random_size()
            outdim = random_size()
            coeff = np.random.randn(indim, outdim)
            model = UpdatingModel(indim, outdim)
            inp = np.random.randn(SAMPLES, indim)
            out = coeff.T.dot(inp.T).T
            model.add_samples(inp, out)
            diff = sum(sum(abs(model.coeff.T - coeff)))
            self.assertTrue(diff < DIFF)

    def test_to_matrix(self):
        '''
        Test the to_matrix function with random vectors.
        '''
        for _ in range(ITERATIONS):
            indim = random_size()
            vec = np.random.randn(indim)
            result = to_matrix(vec)
            self.assertTrue(result.shape == (vec.shape[0], 1))
            self.assertTrue((result[:, 0] == vec).any())

    def test_predict(self):
        '''
        Test the predict function.
        '''
        for _ in range(ITERATIONS):
            indim = random_size()
            outdim = random_size()
            coeff = np.random.randn(indim, outdim)
            model = UpdatingModel(indim, outdim)
            inp = np.random.randn(SAMPLES, indim)
            out = coeff.T.dot(inp.T).T
            model.add_samples(inp, out)
            pred = model.predict(inp[0, :])
            self.assertTrue(sum(abs(pred - out[0, :])) < DIFF)
