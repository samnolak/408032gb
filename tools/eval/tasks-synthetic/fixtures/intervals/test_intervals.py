import unittest

from intervals import merge_intervals


class IntervalsTest(unittest.TestCase):
    def test_merge(self):
        self.assertEqual(merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]), [[1, 6], [8, 10], [15, 18]])

    def test_touching(self):
        self.assertEqual(merge_intervals([[1, 4], [4, 5]]), [[1, 5]])

    def test_unsorted(self):
        self.assertEqual(merge_intervals([[5, 7], [1, 2]]), [[1, 2], [5, 7]])

    def test_empty(self):
        self.assertEqual(merge_intervals([]), [])

    def test_bad_interval(self):
        with self.assertRaises(ValueError):
            merge_intervals([[3, 1]])


if __name__ == "__main__":
    unittest.main()
