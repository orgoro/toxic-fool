from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest


class MyTestCase(unittest.TestCase):
    def test_when_action_then_outcome(self):
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
