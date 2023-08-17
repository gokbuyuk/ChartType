import os
import logging

basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or '^6h0(^gnLB3-mXWJtnS+'
    @staticmethod
    def init_app(app):
        pass

class DevelopmentConfig(Config):
    DEBUG = True
    AWS_REGION = 'us-west-2'
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(name)s in %(module)s %(threadName)s : %(message)s')

class ProductionConfig(Config):
    AWS_REGION = 'us-gov'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s in %(module)s %(threadName)s : %(message)s')


class TestingConfig(Config):
    TESTING = True
    AWS_REGION = 'us-west-2'   

config = {
    'dev'       : DevelopmentConfig,
    'prod'      : ProductionConfig,
    'test'      : TestingConfig,
    'default'   : DevelopmentConfig 
}