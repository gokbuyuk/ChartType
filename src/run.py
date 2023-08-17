import os
from app import create_app
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '../.env')

#load env variables from .env file
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

#get FLASK_CONFIG from env and create an app based on the config
app = create_app(os.getenv('FLASK_CONFIG') or 'default')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)


@app.cli.command()
def test():
    """Run the unit tests."""
    import unittest
    tests = unittest.TestLoader().discover('tests')
    unittest.TextTestRunner(verbosity=2).run(tests)
