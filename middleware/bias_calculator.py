import logging

logger = logging.getLogger(__name__)

UNBIASED, BIASED, STRONGLY_BIASED = 'UNBIASED', "BIASED", 'STRONGLY_BIASED'

class BiasCalculator:
    def __init__(self):
        pass 

    def calculate(self, predictions):
        if predictions['stance']['error'] is not None and \
            predictions['hatespeech']['error'] is None and \
            predictions['hyperpartisan']['error'] is None:
            return self.__only_hatespeech_and_hyperpartisan(predictions)
        
        if predictions['hatespeech']['error'] is not None:
            raise Exception('Hatespeech detector failed! Nested error: ' + predictions['hatespeech']['error'])
        if predictions['hyperpartisan']['error'] is not None:
            raise Exception('Hatespeech detector failed! Nested error: ' + predictions['hyperpartisan']['error'])
        
        if self.__is_strongly_biased(predictions):
            return STRONGLY_BIASED

        if self.__is_biased(predictions):
            return BIASED
        
        return UNBIASED

    def __only_hatespeech_and_hyperpartisan(self, predictions):
        if predictions['hatespeech']['prediction'] != 'NORMAL' and predictions['hyperpartisan']['prediction'] == 'HYPERPARTISAN':
            return STRONGLY_BIASED
        if predictions['hatespeech']['prediction'] != 'NORMAL' or predictions['hyperpartisan']['prediction'] != 'NORMAL':
            return BIASED
        return UNBIASED

    def __is_strongly_biased(self, predictions):
        return predictions['hatespeech']['prediction'] != 'NORMAL' and \
                predictions['hyperpartisan']['prediction'] == 'HYPERPARTISAN' and \
                predictions['stance']['prediction'] != 'DISCUSS'

    def __is_biased(self, predictions):
        return predictions['hatespeech']['prediction'] != 'NORMAL' or \
                predictions['hyperpartisan']['prediction'] == 'HYPERPARTISAN' or \
                predictions['stance']['prediction'] != 'DISCUSS'
