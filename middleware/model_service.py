import requests
import concurrent.futures
from data import new_detector, new_detector_resp, new_preds

DETECTORS = [
    new_detector('hatespeech','http://hatespeech:5100/predict'),
    new_detector('hyperpartisan','http://hyperpartisan:5200/predict'),
    new_detector('stance', 'http://stance:5300/predict'),
]

class DetectorClient:
    def post(self, detector, article):
        try:
            response = requests.post(detector['url'], json=article)
            if response.status_code == requests.codes.ok:
                content = response.json()
                return new_detector_resp(detector['name'], content['prediction'])
            elif response.status_code == requests.codes.bad:
                content = response.json()
                if content is not None:
                  return new_detector_resp(None, None, content['error'])
            return DetectorResponse(error=f'Unknown error(status_code={response.status_code})')
        except Exception as e:
            return new_detector_resp(None, None, f'Exception raised(msg={str(e)})')
        return new_detector_resp(None, None, 'Unknown error')


class ModelService:
    def __init__(self, detectors=None, detector_client=None):
        self.detectors = detectors if detectors is not None else DETECTORS.copy()
        self.detector_client = detector_client if detector_client is not None else DetectorClient()

    def predict(self, article):
        predictions = new_preds(None, None, None)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_detector = {executor.submit(self.detector_client.post, detector, article): detector for detector in self.detectors}
            for future in concurrent.futures.as_completed(future_to_detector):
                detector = future_to_detector[future]
                response = future.result()
                predictions[detector['name']] = response
        return predictions
