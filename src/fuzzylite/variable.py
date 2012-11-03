'''
Created on 27/10/2012

@author: jcrada
'''
from fuzzylite.operator import Operator
from fuzzylite.operator import FuzzyOr
from fuzzylite.term import Cumulative
from collections import OrderedDict
from fuzzylite.defuzzifier import CenterOfGravity


class Variable(object):
    '''Represents a linguistic variable which contains different linguistic terms.'''


    def __init__(self, name):
        self.name = name
        self.terms = OrderedDict()
    
    def minimum(self):
        key = next(iter(self.terms)) #first element
        return self.terms[key].minimum
        
    def maximum(self):
        key = next(reversed(self.terms))
        return self.terms[key].maximum
    
    def fuzzify(self, crisp,  fop=Operator.default()):
        memberships = [str(term.membership(crisp,  fop)) + '/' + term.name \
                            for term in self.terms.values()]
        return ' + '.join(memberships)
    
    def toFCL(self):
        return '\n'.join([self.terms[key].toFCL() for key in self.terms])

class InputVariable(Variable):
    '''Defines a linguistic variable for input.'''
    
    def __init__(self, name):
        Variable.__init__(self, name)
        self.input = float(0.0)
        
    def toFCL(self):
        fcl = []
        fcl.append('FUZZIFY %s' % self.name)
        fcl.append(Variable.toFCL(self))
        fcl.append('END_FUZZIFY')
        return '\n'.join(fcl)

class OutputVariable(Variable):
    '''Defines a linguistic variable for output.'''
    
    def __init__(self, name, defuzzifier = CenterOfGravity(), 
                 accumulate = FuzzyOr.Max):
        Variable.__init__(self, name)
        self.default = None
        self.defuzzifier = defuzzifier
        self.output = Cumulative('output', accumulate)
    
    def toFCL(self):
        fcl = []
        fcl.append('DEFUZZIFY %s' % self.name)
        fcl.append(Variable.toFCL(self))
        fcl.append('ACCU : %s' % self.output.accumulate.__name__.upper())
        fcl.append('METHOD : %s' % self.defuzzifier.toFCL())
        if self.default is not None:
            fcl.append('DEFAULT : %f' % self.default)
        fcl.append('END_DEFUZZIFY')
        return '\n'.join(fcl)
    
    def defuzzify(self):
        
    
if __name__ == '__main__':
#    from collections import OrderedDict
#    d = OrderedDict([('a',1), ('b',2), ('c',3)])
#    print(next(iter(d)))
#    print(next(reversed(d)))
#    for key in d.items():
#        print(key)
    from fuzzylite.term import Triangle
    var = InputVariable('test')
    low = Triangle('Low', 0, 5, 10)
    med = Triangle('Med', 5, 10, 15)
    hi = Triangle('Hi', 10, 15, 20)
    var.terms[low.name] = low
    var.terms[med.name] = med
    var.terms[hi.name] = hi
    var.input = 1
    print('min=', var.minimum())
    print('max=', var.maximum())
    x = 0
    while x < 21:
        print(x, '=', var.fuzzify(x))
        x += 0.5
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
