import unittest
import triangle as tf


class test_area_triangle(unittest.TestCase):
    def test_area_triangle(self):
        """ Test triangle area """
        self.assertEqual(tf.area_triangle(0, 0), 0)
        self.assertEqual(tf.area_triangle(1, 1), 0.5 * 1 * 1)
        self.assertEqual(tf.area_triangle(2, 2), 0.5 * 2 * 2)

    def test_values(self):
        """ Check that input >=0 """
        self.assertRaises(ValueError, tf.area_triangle, -5, 1)
        self.assertRaises(ValueError, tf.area_triangle, 1, -5)
        self.assertRaises(ValueError, tf.area_triangle, -5, -5)
