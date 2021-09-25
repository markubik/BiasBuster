from goose3 import Goose
from newspaper import Article
from data import new_article
import logging

logger = logging.getLogger(__name__)

class GooseArticleExtractor:
    def __init__(self):
        self.goose = Goose()

    def extract(self, url):
        article = self.goose.extract(url=url)
        return new_article(article.title, article.cleaned_text)

class NewspaperArticleExtractor:
    def extract(self, url):
        if len(url) == 0:
            raise Exception('Provided URL is empty')
        article = Article(url)
        article.download()
        article.parse()
        title = article.title
        text = article.text
        logger.info(f'Article extracted(url="{url}", header="{title[:20]}..", text="{article.text[:20]}..")')
        if len(article.title) == 0:
            raise Exception("No title detected")
        if(len(article.text) == 0):
            raise Exception("No article detected")
        return new_article(article.title, article.text)
