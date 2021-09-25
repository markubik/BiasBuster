from article_extractor import NewspaperArticleExtractor
from model_service import ModelService
from bias_calculator import BiasCalculator
import threading

class Service:
    def __init__(self, article_extractor=None, model_service=None, bias_calculator=None):
        self.article_extractor = article_extractor if article_extractor is not None else NewspaperArticleExtractor()
        self.model_service = model_service if model_service is not None else ModelService()
        self.bias_calculator = bias_calculator if bias_calculator is not None else BiasCalculator()
        self.lock = threading.Lock()

    def predict(self, url):
        article = self.article_extractor.extract(url)
        with self.lock:
            predictions = self.model_service.predict(article)
        bias = self.bias_calculator.calculate(predictions)
        return {'bias': bias, 'predictions': predictions}
