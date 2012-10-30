'''
Created on 10/10/2012

@author: jcrada
'''

from fuzzylite.fuzzy_operator import FuzzyOperator
from fuzzylite.hedge_dict import HedgeDict
from collections import OrderedDict

class FuzzyEngine:
    '''Wraps the whole system.'''
    

    def __init__(self, name, fuzzy_operator = FuzzyOperator.Default(),
                  hedge = HedgeDict()):
        self.name = name
        self.fuzzy_operator = fuzzy_operator
        self.hedge = hedge
        self.input = OrderedDict()
        self.output = OrderedDict()
    
    def process(self):
        pass
    
    def toFCL(self):
        fcl = ['FUNCTION_BLOCK %s\n\n' % self.name]
        
        fcl.append('VAR_INPUT\n')
        for key in self.input:
            fcl.append('%s: REAL;\n' % key)
        fcl.append('END_VAR\n\n')
        
        for key in self.input:
            fcl.append('FUZZIFY %s\n' % key)
            for key, term in self.input[key].terms.items():
                fcl.append('%s;\n' % term.toFCL())
                
        fcl.append('END_FUZZIFY\n\n')
        
        fcl.append('VAR_OUTPUT\n')
        for key in self.output:
            fcl.append('%s: REAL\n' % key)
        fcl.append('END_VAR\n\n')
        
        for key in self.output:
            fcl.append('DEFUZZIFY %s\n' % key)
            for key, term in self.output[key].terms.items():
                fcl.append('%s\n' % term.toFCL())
        fcl.append('END_DEFUZZIFY\n\n')

#        for (int i = 0; i < numberOfRuleBlocks(); ++i) {
#            ss << ruleBlock(i)->toString() << "\n\n";
#        }

        fcl.append('END_FUNCTION_BLOCK')
        return ''.join(fcl)
    
if __name__ == '__main__':
    print(FuzzyEngine(None).toFCL())
    
        
        