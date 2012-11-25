'''
Created on 10/10/2012

@author: jcrada
'''


from collections import OrderedDict

from fl.operator import FuzzyAnd, FuzzyOr, FuzzyActivation, FuzzyAccumulation
from fl.hedge import HedgeDict
from fl.defuzzifier import CenterOfGravity

class Operator:
    '''
    Operator configures the whole engine. It defines all the operators needed
    in the engine. 
    '''
    
    def __init__(self, name=None, tnorm=FuzzyAnd.Min, snorm=FuzzyOr.Max,
                 activation=FuzzyActivation.Min, accumulation=FuzzyAccumulation.Max,
                 defuzzifier=CenterOfGravity()):
        '''Constructs a Operator with default values.'''
        self.name = name
        self.tnorm = tnorm
        self.snorm = snorm
        self.activation = activation
        self.accumulation = accumulation
        self.defuzzifier = defuzzifier
    
    def __str__(self):
        result = ['Operator %s' % self.name]
        result.append('tnorm = %s' % self.tnorm.__name__)
        result.append('snorm = %s' % self.snorm.__name__)
        result.append('activation = %s' % self.activation.__name__)
        result.append('accumulation = %s' % self.accumulation.__name__)
        result.append('defuzzifier = %s' % self.defuzzifier)
        return '\n'.join(result)

class Engine:
    '''A fuzzy logic engine.'''
    

    def __init__(self, name = None):
        self.name = name
        self.operator = None
        self.hedge =  HedgeDict()
        self.input = OrderedDict()
        self.output = OrderedDict()
        self.ruleblock = OrderedDict()
    
    def configure(self, fop):
        self.operator = fop
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
    from fl.example import Example
    fe = Example.simple_mamdani()
    
    
        
        