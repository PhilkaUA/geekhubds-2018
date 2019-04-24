import re

from tokenizer import TextProcessor
from words_test import Word


def main(text: str):
    '''main program for tasks'''

    # split by words
    _text = TextProcessor(text)

    print('Input text:', text)
    print('Output text:', _text.tokenize(text))

    # test for each word
    _words_test = Word()

    result = []
    for word in _text.tokenize(text):
        result.append(
            (word, (_words_test.is_numerical(word), _words_test.is_range(word),\
                    _words_test.is_stopword(word))))

    print('\nOutput test: [token, [is_numerical, is_range, is_stopword]]')
    print('\nResult:', result)


if __name__ == '__main__':
    text = 'Some text. 33-11 . Improved own?, provided - blessing may peculiar domestic.'
    main(text)