from flask import Flask
from config import config
from app.service import main as service_blueprint

def create_app(config_name):
    """Create a flask app based on config matching config_name in config.py"""
    app = Flask(__name__)
    app.config.from_object(config[config_name]) 
    config[config_name].init_app(app)
    app.register_blueprint(service_blueprint)

    return app
