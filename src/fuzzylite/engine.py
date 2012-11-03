'''
Created on 10/10/2012

@author: jcrada
'''

from fuzzylite.hedge_dict import HedgeDict
from collections import OrderedDict

class Engine:
    '''Wraps the whole system.'''
    

    def __init__(self, name, fop,
                  hedge = HedgeDict()):
        self.name = name
        self.fop = fop
        self.hedge = hedge
        self.input = OrderedDict()
        self.output = OrderedDict()
        self.ruleblock = OrderedDict()
    
    def process(self):
        if len(self.output) == 0:
            raise ValueError('engine has no outputs')
        if len(self.ruleblock) == 0:
            raise ValueError('engine has no ruleblocks')
        for key in self.output:
            self.output[key].output.clear()
        for key in self.ruleblock:
            self.ruleblock[key].fire_rules()
    
    def toFCL(self):
        fcl = ['FUNCTION_BLOCK %s' % self.name]
        fcl.append('')
        
        fcl.append('VAR_INPUT')
        for key in self.input:
            fcl.append('%s: REAL;' % key)
        fcl.append('END_VAR')
        fcl.append('')
        
        fcl.append('VAR_OUTPUT')
        for key in self.output:
            fcl.append('%s: REAL;' % key)
        fcl.append('END_VAR')
        fcl.append('')
        
        for key in self.input:
            fcl.append(self.input[key].toFCL())
        fcl.append('')
        
        for key in self.output:
            fcl.append(self.output[key].toFCL())
        fcl.append('')    

        for key in self.ruleblock:
            fcl.append(self.ruleblock[key].toFCL())
        fcl.append('')

        fcl.append('END_FUNCTION_BLOCK')
        return '\n'.join(fcl)
    
if __name__ == '__main__':
    e = Engine(None, None)
    from fuzzylite.example import Example
    fe = Example.simple_mamdani()
    print (fe.toFCL())
    
        
        