'''
Created on 3/11/2012

@author: jcrada
'''
from fuzzylite.operator import Operator
class RuleBlock(list):
    '''
    A set of rules.
    '''

    def __init__(self, name):
        self.name = name
        self.fop = None
        
    def configure(self, fop):
        self.fop = fop
    
    def fire_rules(self):
        if len(self) == 0: 
            raise ValueError('no rules to fire')
        for rule in self:
            strength = rule.firing_strength(self.fop)
            if strength > 0.0:
                rule.fire(strength, self.fop.activation)
    
if __name__ == '__main__':
    from fuzzylite.operator import Operator
    x = RuleBlock('a')
    x.configure(Operator.default())
    
    
