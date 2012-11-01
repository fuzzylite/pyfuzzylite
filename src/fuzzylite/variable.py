'''
Created on 27/10/2012

@author: jcrada
'''
from fuzzylite.membership_function import Composite
from collections import OrderedDict

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
    
    def compound(self):
        terms = list(self.terms.values())
        return Composite(self.name, terms)
    
    def fuzzify(self, crisp):
        memberships = [str(term.membership(crisp)) + '/' + term.name \
                            for term in self.terms.values()]
        return ' + '.join(memberships)
    

class InputVariable(Variable):
    '''Defines a linguistic variable for input.'''
    
    def __init__(self, name):
        Variable.__init__(self, name)
        self.input = float(0.0)

class OutputVariable(Variable):
    '''Defines a linguistic variable for output.'''
    
    def __init__(self, name):
        Variable.__init__(self, name)
        self.output = Composite('output')

if __name__ == '__main__':
#    from collections import OrderedDict
#    d = OrderedDict([('a',1), ('b',2), ('c',3)])
#    print(next(iter(d)))
#    print(next(reversed(d)))
#    for key in d.items():
#        print(key)
    from fuzzylite.membership_function import Triangular
    var = InputVariable('test')
    low = Triangular('Low', 0, 5, 10)
    med = Triangular('Med', 5, 10, 15)
    hi = Triangular('Hi', 10, 15, 20)
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
