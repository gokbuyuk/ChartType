import unittest
from flask import current_app
from app import create_app


class BasicTestCase(unittest.TestCase):
    def init_app(self):
        self.app = create_app('test')

    def test_app_exists(self):
        self.assertFalse(current_app is None)

    def test_app_is_testing(self):
        self.assertTrue(current_app.config['TESTING'])