import unittest

from inventory.stock import OutOfStock, Stock


class StockTest(unittest.TestCase):
    def test_add_remove(self):
        s = Stock()
        s.add("A", 5)
        s.remove("A", 2)
        self.assertEqual(s.available("A"), 3)

    def test_remove_everything(self):
        s = Stock()
        s.add("A", 2)
        s.remove("A", 2)
        self.assertEqual(s.available("A"), 0)

    def test_remove_too_many(self):
        s = Stock()
        s.add("A", 2)
        with self.assertRaises(OutOfStock):
            s.remove("A", 3)
        self.assertEqual(s.available("A"), 2)

    def test_unknown_sku(self):
        with self.assertRaises(OutOfStock):
            Stock().remove("B", 1)


if __name__ == "__main__":
    unittest.main()
