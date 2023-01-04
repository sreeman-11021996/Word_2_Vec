from layers import dense
import numpy as np
import unittest

class Layer_Test(unittest.TestCase):

    def setUp(self)->None:
        self._num_units = 10
        self._layer = dense.Dense(units=self._num_units)

    def test_dense_output_shape(self):
        self.assertEqual(self._layer(np.random.randn(8, 5)).shape, \
            (self._num_units, 5))
        

if __name__ == "__main__":
    unittest.main()