import unittest
import numpy as np

from ai_code import label_encode,one_hot_encode,minkowski_distance,dot,norm,mean,variance,std
from my_code import label_encode,one_hot_encode,minkowski_distance,dot,norm,mean,variance,std

class TestEncoding(unittest.TestCase):

    def test_label_encode(self):
        data = ['Red', 'Blue', 'Green', 'Blue', 'Red']

        enc, mp = label_encode(data)

        self.assertEqual(enc, [0, 1, 2, 1, 0])
        self.assertEqual(mp, {'Red': 0, 'Blue': 1, 'Green': 2})

    def test_one_hot_encode(self):
        data = ['Red', 'Blue', 'Green', 'Blue', 'Red']

        result, mp = one_hot_encode(data)

        expected = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ]

        self.assertEqual(result, expected)
        self.assertEqual(mp, {'Red': 0, 'Blue': 1, 'Green': 2})


class TestDistanceAndVectorFunctions(unittest.TestCase):

    def test_minkowski_distance(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 6, 8])

        result = minkowski_distance(a, b, 2)

        self.assertAlmostEqual(result, 7.0710678119, places=5)

    def test_dot_product(self):
        A = [1, 2, 3]
        B = [4, 5, 6]

        result = dot(A, B)

        self.assertEqual(result, 32)

    def test_norm(self):
        V = [3, 4]

        result = norm(V)

        self.assertEqual(result, 5)


class TestStatistics(unittest.TestCase):

    def test_mean(self):
        data = [10, 20, 30, 40, 50]

        result = mean(data)

        self.assertEqual(result, 30)

    def test_variance(self):
        data = [10, 20, 30, 40, 50]

        result = variance(data)

        self.assertEqual(result, 200)

    def test_standard_deviation(self):
        data = [10, 20, 30, 40, 50]

        result = std(data)

        self.assertAlmostEqual(result, 14.1421356237, places=5)


if __name__ == '__main__':
    unittest.main()