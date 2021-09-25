from service import Service
from model_service import ModelService
from data import new_detector_resp, new_preds

class DetectorClientMock:
    def post(self, detector, data):
        return new_detector_resp(detector['name'], 'FOO')

def test_service():
    detector_client_mock = DetectorClientMock()
    model_service = ModelService(detector_client=DetectorClientMock())
    service = Service(model_service=model_service)
    url = 'https://techcrunch.com/2021/09/22/heres-whats-happening-on-day-two-at-techcrunch-disrupt-2021/'
    expected = {
        'bias': 'BIASED',
        'predictions': new_preds(
            new_detector_resp('hatespeech', 'FOO'),
            new_detector_resp('hyperpartisan', 'FOO'),
            new_detector_resp('stance', 'FOO'),
        )
    }

    response = service.predict(url)

    print(response)
    assert(response == expected)
    print('PASSED')


if __name__ == '__main__':
    test_service()