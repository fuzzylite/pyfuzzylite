'''
Created on 3/11/2012

@author: jcrada
'''

from fuzzylite.operator import Operator

class RuleBlock(list):
    '''
    A set of rules.
    '''


    def __init__(self, name=None, fop=Operator.default()):
        self.name = name
        self.fop = fop
        
    def fire_rules(self):
        for rule in self:
            rule.fire(rule.firing_strength(self.fop), self.fop)
    
    def toFCL(self):
        fcl = ['RULEBLOCK %s\n' % self.name]
        for i in range(0, len(self)):
            fcl.append('  RULE %i: %s;\n' % (i + 1, str(self[i])))
            
        fcl.append('END_RULEBLOCK')
        
        return ' '.join(fcl)
    
    
if __name__ == '__main__':
    pass
