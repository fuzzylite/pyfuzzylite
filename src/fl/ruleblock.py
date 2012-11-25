'''
Created on 3/11/2012

@author: jcrada
'''

class RuleBlock(list):
    '''
    A set of rules.
    '''

    def __init__(self, name = None):
        self.name = name
        self.tnorm = None
        self.snorm = None
        self.activation = None
        
    def configure(self, fop):
        self.tnorm = fop.tnorm
        self.snorm = fop.snorm
        self.activation = fop.activation
    
    def fire_rules(self):
        if len(self) == 0: 
            raise ValueError('no rules to fire')
        for rule in self:
            strength = rule.firing_strength(self.tnorm, self.snorm)
            if strength > 0.0:
                rule.fire(strength, self.activation)
    
if __name__ == '__main__':
    from fl.engine import Operator
    x = RuleBlock('a')
    fop = Operator()
    x.configure(fop)
    print(x.tnorm)
    from fl.operator import FuzzyOr
    fop.tnorm = FuzzyOr.Max
    print(x.tnorm)
    
    
    
