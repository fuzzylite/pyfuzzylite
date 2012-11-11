'''
Created on 27/10/2012

@author: jcrada
'''

from collections import OrderedDict

class Variable(object):
    '''A Linguistic Variable.
    
    Represents a fuzzy linguistic variable such as Energy, Health, Service, Tip.
    This variable assumes that terms are added in proper order.
    
    Attributes:
        name: the name of this variable.
        term: an ordered dictionary of fuzzy terms.'''

    def __init__(self, name):
        self.name = name
        self.term = OrderedDict()
    
    def __iter__(self):
        '''Returns a generator that iterates through all the terms of this variable.''' 
        for key in self.term:
            yield self.term[key]
    
    def configure(self, fop): 
        '''Configures the variable using the definitions from FuzzyOperator.'''
        pass
    
    def minimum(self):
        '''Returns the minimum value from the first term added to this variable.'''
        key = next(iter(self.term)) #first element
        return self.term[key].minimum
        
    def maximum(self):
        '''Returns the maximum value from the last term added to this variable.'''
        key = next(reversed(self.term))
        return self.term[key].maximum
    
    def fuzzify(self, x):
        '''Returns a string defining the degrees of membership of x to each term.'''
        fuzzy = []
        for term in self:
            if x is not None:
                fuzzy.append('%f/%s' % (term.membership(x), term.name))
            else:
                fuzzy.append('None/%s' % term.name)
        return ' + '.join(fuzzy)
    
    def toFCL(self):
        '''Returns a string representing this variable in the Fuzzy Control Language.''' 
        return '\n'.join([term.toFCL() for term in self])


        

class InputVariable(Variable):
    '''An input variable.
    
    Defines input variables such as Energy or Service.
    
    Attributes:
        input: a float defining the input value of this variable.'''
    
    def __init__(self, name):
        Variable.__init__(self, name)
        self.input = float(0.0)
    
    def toFCL(self):
        fcl = []
        fcl.append('FUZZIFY %s' % self.name)
        fcl.append(Variable.toFCL(self))
        fcl.append('END_FUZZIFY')
        return '\n'.join(fcl)


from fuzzylite.term import Cumulative

class OutputVariable(Variable):
    '''An output varible such as Health or Tip.
    
    Defines a linguistic variable for output.
    
    Attributes:
        default: a float value to assume by default if there is no output.
        defuzzifier: an instance of a defuzzifier method.
        output: a Cumulative term to which Output terms will be appended.
    '''
    
    def __init__(self, name, default = None):
        Variable.__init__(self, name)
        self.default = default
        self.defuzzifier = None
        self.output = Cumulative('output')
    
    def configure(self, fop):
        Variable.configure(self, fop)
        self.defuzzifier = fop.defuzzifier
        self.output.accumulation = fop.accumulation
    
    def toFCL(self):
        fcl = []
        fcl.append('DEFUZZIFY %s' % self.name)
        fcl.append(Variable.toFCL(self))
        fcl.append('METHOD : %s' % self.defuzzifier.toFCL())
        if self.default is not None:
            fcl.append('DEFAULT : %f' % self.default)
        fcl.append('END_DEFUZZIFY')
        return '\n'.join(fcl)
    
    def defuzzify(self):
        '''Returns a single float value representing the defuzzified output.'''
        if self.output.is_empty():
            return self.default
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
