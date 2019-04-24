import re

class TextProcessor():
    '''Tokenize texts by spaces'''

    def __init__(self, text: str):
        self.text = text

    def tokenize(self, text):
        '''Tokenize texts by sybmols'''
        patern = re.compile(r"[.,:;?! ]")
        return list(filter(bool, re.split(patern, text)))