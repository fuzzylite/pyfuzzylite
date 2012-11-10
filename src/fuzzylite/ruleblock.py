'''
Created on 3/11/2012

@author: jcrada
'''
from fuzzylite.rule import Rule
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
    
    def toFCL(self):
        fcl = []
        fcl.append('RULEBLOCK %s' % self.name)
        fcl.append('%s : %s;' % (Rule.FR_AND.upper(),
                                       self.fop.tnorm.__name__.upper()))
        fcl.append('%s : %s;' % (Rule.FR_OR.upper(),
                                       self.fop.snorm.__name__.upper()))
        fcl.append('ACT : %s;' % self.fop.activation.__name__.upper())
        fcl.append('ACCU : %s;' % self.fop.accumulation.__name__.upper())
        fcl.append('')
        for i in range(0, len(self)):
            fcl.append('RULE %i: %s;' % (i + 1, str(self[i])))
            
        fcl.append('END_RULEBLOCK')
        
        return '\n'.join(fcl)
    
    
if __name__ == '__main__':
    from fuzzylite.operator import Operator
    x = RuleBlock('a')
    x.configure(Operator.default())
    print(x.toFCL())
    
