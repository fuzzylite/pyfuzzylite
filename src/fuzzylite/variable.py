'''
Created on 27/10/2012

@author: jcrada
'''
from fuzzylite.term import Output
from collections import OrderedDict


class Variable(object):
    '''Represents a linguistic variable which contains different linguistic term.'''


    def __init__(self, name):
        self.name = name
        self.term = OrderedDict()
    
    def __iter__(self):
        for key in self.term:
            yield self.term[key]
    
    def configure(self, fop): 
        for name in self.term:
            self.term[name].configure(fop)
    
    def minimum(self):
        key = next(iter(self.term)) #first element
        return self.term[key].minimum
        
    def maximum(self):
        key = next(reversed(self.term))
        return self.term[key].maximum
    
    def fuzzify(self, crisp):
        fuzzy = ['%f/%s' % (term.membership(crisp), term.name) for term in self]
        return ' + '.join(fuzzy)
    
    def toFCL(self):
        return '\n'.join([term.toFCL() for term in self])


        

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
    
    def __init__(self, name):
        Variable.__init__(self, name)
        self.default = None
        self.defuzzifier = None
        self.output = Output('output')
    
    def configure(self, fop):
        Variable.configure(self, fop)
        self.defuzzifier = fop.defuzzifier
        self.output.configure(fop)
    
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
        return self.defuzzifier.defuzzify(self.output)
    
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
    var.term[low.name] = low
    var.term[med.name] = med
    var.term[hi.name] = hi
    var.input = 1
    print('min=', var.minimum())
    print('max=', var.maximum())
    x = 0
    while x < 21:
        print(x, '=', var.fuzzify(x))
        x += 0.5
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
