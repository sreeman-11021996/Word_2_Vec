from activation import activations
import numpy as np
import unittest

class Activation_Test(unittest.TestCase):
    
    def test_relu(self):
        _input = np.array([-1, 1])
        _output = np.array([0, 1])

        relu = activations.ReLU()

        # Checking shapes
        self.assertEqual(relu(np.random.randn(10, 20)).shape, (10, 20))
        self.assertEqual(relu.d_activation(np.random.randn(10, 20)).shape,(10, 20))
        
        # Checking values
        self.assertEqual(relu(_input).tolist(), _output.tolist())
        self.assertEqual(relu.d_activation(_input).tolist(), _output.tolist())
    
    def test_sigmoid(self):
        _input = np.array([-100, 0, 100])
        
        sigmoid = activations.Sigmoid()
        
        # Checking shapes
        self.assertEqual(sigmoid(np.random.randn(10, 20)).shape, (10, 20))
        self.assertEqual(sigmoid.d_activation(np.random.randn(10, 20)).shape,
                         (10, 20))
        
        # Checking values
        self.assertEqual(round(sigmoid(_input[0]),2), 0.0)
        self.assertEqual(round(sigmoid(_input[1]),2), 0.5)
        self.assertEqual(round(sigmoid(_input[2]),2), 1.0)
        self.assertEqual(round(sigmoid.d_activation(_input[0]),2), 0.0)
        self.assertEqual(round(sigmoid.d_activation(_input[2]),2), 0.0)
        