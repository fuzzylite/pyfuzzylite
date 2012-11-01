'''
Created on 27/10/2012

@author: jcrada
'''

from fuzzylite.fuzzy_operator import FuzzyOperator
class FuzzyRule:
    '''Defines a fuzzy rule'''
    
    FR_IF = 'if'
    FR_IS = 'is'
    FR_THEN = 'then'
    FR_AND = 'and'
    FR_OR = 'or'
    FR_WITH = 'with'

    def __init__(self):
        self.antecedent = None
        self.consequent = None
        pass
    
    def firing_strength(self, fop = FuzzyOperator.default()):
        return self.antecedent.firing_strength(fop)
    
    def fire(self, strength=None, fop = FuzzyOperator.default()):
        if strength is None: 
            strength = self.firing_strength(fop = fop)
        self.consequent.fire(strength)
        

class FuzzyAntecedent(object):
    
    def firing_strength(self, fop = FuzzyOperator.default()):
        raise NotImplementedError('firing_strength')

class FuzzyConsequent(object):
    
    def fire(self, strength):
        raise NotImplementedError('fire')

