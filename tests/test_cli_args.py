import unittest
import argparse
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.simulate import parse_arguments
from unittest.mock import patch

class TestCLIArgs(unittest.TestCase):
    @patch('sys.argv', ['simulate.py', '--method', 'RK45', '--target-mass', '600.0', '--edt-length', '1000.0', '--inclination', '0.0'])
    def test_cli_args(self):
        args = parse_arguments()
        self.assertEqual(args.target_mass, 600.0)
        self.assertEqual(args.edt_length, 1000.0)
        self.assertEqual(args.inclination, 0.0)

if __name__ == '__main__':
    unittest.main()

