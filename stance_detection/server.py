from flask import Flask, request, jsonify
from model import StanceDetectionModel
import time
import traceback
import logging
import sys
import threading
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
model = StanceDetectionModel()
app = Flask(__name__)

lock = threading.Lock()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        content = request.get_json(silent=True)
        logger.debug('Received request with:' + str(content))
        head = content['header']
        body = content['text']
        with lock:
            prediction = model.predict(head, body)
        return jsonify({'prediction': prediction})
    except Exception as e:
        logger.error('Failed to make a prediction')
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    port = sys.argv[1] if sys.argv.__len__() > 1 else 5300

    logger.info('Loading model...')
    start_time = time.time()
    model.load()
    end_time = time.time()
    logger.info('Model loaded in ' + str(end_time-start_time) + ' seconds')

    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=port)
