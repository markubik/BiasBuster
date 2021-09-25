from flask import Flask, request, jsonify
import traceback
from service import Service
import logging
import sys
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
app = Flask(__name__)



service = Service()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        content = request.get_json(silent=True)
        response = service.predict(content['url'])
        return jsonify(response)
    except Exception as e:
        logger.error('Failed to make a prediction')
        logger.error(traceback.format_exc())
        return str(e), 400


if __name__ == '__main__':
    port = sys.argv[1] if sys.argv.__len__() > 1 else 5000

    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=port)
