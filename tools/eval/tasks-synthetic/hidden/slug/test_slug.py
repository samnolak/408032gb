import unittest

from slug import slugify


class SlugTest(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(slugify("Hello, World!"), "hello-world")

    def test_spaces(self):
        self.assertEqual(slugify("  many   spaces here "), "many-spaces-here")

    def test_digits(self):
        self.assertEqual(slugify("Release 2.0 notes"), "release-2-0-notes")

    def test_nothing_left(self):
        self.assertEqual(slugify("!!!"), "")

    def test_accents(self):
        self.assertEqual(slugify("Crème brûlée"), "creme-brulee")


if __name__ == "__main__":
    unittest.main()
