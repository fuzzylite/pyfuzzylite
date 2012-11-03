'''
Created on 3/11/2012

@author: jcrada
'''

class RuleBlock(list):
    '''
    A set of rules.
    '''


    def __init__(self, name, fop):
        self.name = name
        self.fop = fop
        
    def fire_rules(self):
        if len(self) == 0: 
            raise ValueError('no rules to fire')
        for rule in self:
            rule.fire(rule.firing_strength(self.fop), self.fop)
    
    def toFCL(self):
        fcl = []
        fcl.append('RULEBLOCK %s' % self.name)
        fcl.append(self.fop.toFCL())
        for i in range(0, len(self)):
            fcl.append('RULE %i: %s;' % (i + 1, str(self[i])))
            
        fcl.append('END_RULEBLOCK')
        
        return '\n'.join(fcl)
    
    
if __name__ == '__main__':
    from fuzzylite.operator import Operator
    RuleBlock('',Operator.default()).fire_rules()
