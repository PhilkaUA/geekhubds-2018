import re

class Word:
    '''Check word for properties'''

    def is_numerical(self, word):
        '''Numerical test'''
        return word.isnumeric()

    def is_range(self, word):
        '''Range test for format NN-NN | NN/NN'''
        match = re.fullmatch(r'\d+[/-]\d+', word)
        return ('True' if match else 'False')

    def is_stopword(self, word):
        '''Test for stopwords'''
        stopwords = ['am', 'and', 'of', 'some']
        return word in stopwords