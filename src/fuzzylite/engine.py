'''
Created on 10/10/2012

@author: jcrada
'''

from fuzzylite.hedge_dict import HedgeDict
from fuzzylite.operator import Operator
from fuzzylite.ruleblock import RuleBlock
from collections import OrderedDict

class Engine:
    '''Wraps the whole system.'''
    

    def __init__(self, name, fop = Operator.default(),
                  hedge = HedgeDict()):
        self.name = name
        self.fop = fop
        self.hedge = hedge
        self.input = OrderedDict()
        self.output = OrderedDict()
        self.ruleblock = OrderedDict()
    
    def process(self):
        for key in self.output:
            self.output[key].output.clear()
        
    
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

        for key in self.ruleblock:
            fcl.append('%s\n\n' % self.ruleblock[key].toFCL())


        fcl.append('END_FUNCTION_BLOCK')
        return ''.join(fcl)
    
if __name__ == '__main__':
    print(Engine(None).toFCL())
    
        
        