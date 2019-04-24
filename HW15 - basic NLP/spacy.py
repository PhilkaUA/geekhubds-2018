import itertools
import spacy

#!python -m spacy download en_core_web_sm
#nlp = spacy.load('en_core_web_md')
nlp = spacy.load("en_core_web_sm")

doc = nlp(u'Pt placed in LL first position, O2 by mask in placed, IV bolus running.\
Prolonged late decel x 1 noted, now returned to baseline 150 bpm with mod variability and pos accels.')

class MyMatcher():

    def __init__(self, doc):
        self.doc = doc

    def _sentences(self, doc):
        '''split to sentences'''
        return list(doc.sents)

    def _pairwise(self, iterable):
        '''pairwise list elements
        s -> (s0,s1), (s1,s2), (s2, s3)
        needs - itertools'''
        self.iterable = iterable
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

    def remove_pucnt_pairs(self, doc):
        '''remove punctuaton and pairwise list elements'''
        _pros = []
        _text = []
        # print(self._sentences(doc))
        for sent in self._sentences(doc):
            for token in sent:
                if token.pos_ != 'PUNCT':
                    _pros.append(token.pos_)
                    _text.append(token.text)

        return zip(self._pairwise(_pros), self._pairwise(_text))

    def main(self, doc):
        '''results output'''

        for pros, text in self.remove_pucnt_pairs(doc):

            if pros == ('NOUN', 'VERB'):
                print(text)
            elif pros == ('VERB', 'NOUN'):
                print(text)
            elif pros == ('NOUN', 'NUM'):
                print(text)
            elif pros == ('VERB', 'NUM'):
                print(text)
            elif pros == ('NUM', 'NOUN'):
                print(text)
            elif pros == ('NUM', 'VERB'):
                print(text)

matcher = MyMatcher(doc)
matcher.main(doc)

##
## not the python way solution and output as you expected, but the result is the same ^^
##

# gold_items = {
#     '1 noted',
#     'baseline 150',
#     'bolus running',
#     'pt placed',
# }
# gold_text = """Pt placed in LL first position, O2 by mask in placed, IV bolus running.
# Prolonged late decel x 1 noted, now returned to baseline 150 bpm with mod variability and pos accels."""
#
# matcher = MyMatcher()
# cand_items = set((x.casefold() for x in matcher(gold_text)))
# assert cand_items == gold_items