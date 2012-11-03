'''
Created on 27/10/2012

@author: jcrada
'''

from fuzzylite.operator import Operator
class Rule(object):
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
    
    def firing_strength(self, fop = Operator.default()):
        return self.antecedent.firing_strength(fop)
    
    def fire(self, strength, fop = Operator.default()):
        self.consequent.fire(strength)

    def __str__(self):
        return '%s %s %s %s' % (Rule.FR_IF, str(self.antecedent), 
                                Rule.FR_THEN, str(self.consequent))

class FuzzyAntecedent(object):
    
    def firing_strength(self, fop):
        raise NotImplementedError('firing_strength')

class FuzzyConsequent(object):
    
    def fire(self, strength):
        raise NotImplementedError('fire')

