from app.service import main
from flask import jsonify
from log import ApiLog
import logging
from flask_cors import CORS

logger = logging.getLogger(__name__)
CORS(main)


@main.route('/')
def welcome():
    """
    Heartbeat message for Loadbalancer 
    ---
    """
    data={'mlaas':'hello'}
    return jsonify(data), 200

@main.route('/status', methods=['GET'])
def status():
    """
    Heartbeat message for Loadbalancer
    """
    
    data={'mlaas':'running'}
    logger.info(data)
    return jsonify(data), 200

