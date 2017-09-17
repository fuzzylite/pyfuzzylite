'''
Created on 27/10/2012

@author: jcrada
'''
import logging


class Rule(object):
    '''Defines a fuzzy rule'''

    FR_IF = 'if'
    FR_IS = 'is'
    FR_THEN = 'then'
    FR_AND = 'and'
    FR_OR = 'or'
    FR_WITH = 'with'

    def __init__(self, text: str):
        self.antecedent = None
        self.consequent = None
        self.text = text
        self.logger = logging.getLogger(type(self).__name__)

    def configure(self, fop):
        pass

    def firing_strength(self, tnorm, snorm):
        return self.antecedent.firing_strength(tnorm, snorm)

    def fire(self, strength, activation):
        self.consequent.fire(strength, activation)

    def __str__(self):
        return '%s %s %s %s' % (Rule.FR_IF, str(self.antecedent),
                                Rule.FR_THEN, str(self.consequent))


class FuzzyAntecedent(object):
    def __init__(self):
        self.logger = logging.getLogger(type(self).__name__)

    def firing_strength(self, tnorm, snorm):
        raise NotImplementedError('firing_strength')


class FuzzyConsequent(object):
    def __init__(self):
        self.logger = logging.getLogger(type(self).__name__)

    def fire(self, strength, activation):
        raise NotImplementedError('fire')
