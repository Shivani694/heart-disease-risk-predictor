import unittest
from app import app
import json

class FlaskAppTests(unittest.TestCase):

    def setUp(self):
        # Create a test client
        self.app = app.test_client()
        # Propagate exceptions to the test client
        self.app.testing = True

    def test_welcome_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_login_page(self):
        response = self.app.get('/login')
        self.assertEqual(response.status_code, 200)

    def test_login_post(self):
        response = self.app.post('/login', data=dict(contact="test@example.com"))
        # Should redirect to dashboard (302)
        self.assertEqual(response.status_code, 302)

    def test_predict_without_login(self):
        # We try to post without session
        response = self.app.post('/predict', data=dict(
            age="45", gender="1", bp="130", cholesterol="220", 
            bs="0", hr="80", cp="2", angina="0"
        ))
        # Should redirect back to login
        self.assertEqual(response.status_code, 302)

if __name__ == '__main__':
    unittest.main()
