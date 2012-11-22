'''
Created on 10/10/2012

@author: jcrada
'''

from fuzzylite.hedge_dict import HedgeDict
from fuzzylite.operator import Operator
from collections import OrderedDict

class Engine:
    '''Wraps the whole system.'''
    

    def __init__(self, name = None):
        self.name = name
        self.hedge =  HedgeDict()
        self.input = OrderedDict()
        self.output = OrderedDict()
        self.ruleblock = OrderedDict()
    
    def configure(self, fop = Operator.default()):
        for variable in self.input:
            self.input[variable].configure(fop)
        for variable in self.output:
            self.output[variable].configure(fop)
        for name in self.ruleblock:
            self.ruleblock[name].configure(fop)
        
    
    def process(self):
        if len(self.output) == 0:
            raise ValueError('engine has no outputs')
        if len(self.ruleblock) == 0:
            raise ValueError('engine has no ruleblocks')
        for key in self.output:
            self.output[key].output.clear()
        for key in self.ruleblock:
            self.ruleblock[key].fire_rules()
        
    
if __name__ == '__main__':
    e = Engine()
    from fuzzylite.example import Example
    fe = Example.simple_mamdani()
    
    
        
        