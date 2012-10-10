'''
Created on 10/10/2012

@author: jcrada
'''

from fuzzylite.fuzzy_operator import FuzzyOperator
from fuzzylite.math_utils import Scale


class LinguisticTerm:
    '''
    Provides the necessary properties for any linguistic term
    '''
    
    def __init__(self, name, minimum, maximum, modulation_degree = 1.0, 
                 fuzzy_operator = FuzzyOperator.Default()):
        self.name = name
        self.minimum = minimum
        self.maximum = maximum
        self.modulation_degree = modulation_degree
        self.fuzzy_operator = fuzzy_operator
        
    
    def toFCL(self):
        '''Returns a string in Fuzzy Control Language.'''
        return 'TERM ' + str(self.name) + ' := ' + \
            str(self.__class__.__name__) + ' ';
    
    def membership(self, crisp): raise NotImplementedError()


class Triangular(LinguisticTerm):
    '''
    Defines a triangular term using minimum as the left vertex, 
    middle as the center vertex, and maximum as the right vertex
    '''
    
    def __init__(self, name, minimum, middle_vertex, maximum):
        LinguisticTerm.__init__(self, name, minimum, maximum)
        self.middle_vertex = middle_vertex
    
    def toFCL(self):
        return LinguisticTerm.toFCL(self) + \
        '(' + str(self.minimum) + ', ' + \
        str(self.middle_vertex) + ', ' + str(self.maximum) + ')'

    def membership(self, crisp):
        if crisp < self.minimum or crisp > self.maximum: return 0.0
        mu = 0.0;
        if crisp < self.middle_vertex :
            mu = Scale(crisp, 0.0, 1.0, self.minimum, self.middle_vertex)
        else:
            mu = Scale(crisp, 1.0, 0.0,  self.middle_vertex, self.maximum);
        return self.fuzzy_operator.modulate(mu, self.modulation_degree);

class Rectangular(LinguisticTerm):
    '''Defines a rectangular term in the range [minimum, maximum].'''
    def __init__(self, name, minimum, maximum):
        LinguisticTerm.__init__(self, name, minimum, maximum)
    
    def toFCL(self):
        return LinguisticTerm.toFCL(self) + \
            '(' + str(self.minimum) + ', ' + str(self.maximum) + ')'
    
    def membership(self, crisp):
        mu = None
        if crisp >= self.minimum and crisp <= self.maximum: mu = 1.0
        else: mu = 0.0
        return self.fuzzy_operator.modulate(mu, self.modulation_degree)

class Trapezoidal(LinguisticTerm):
    '''
    Defines a trapezoidal term using minimum and maximum as left-most and 
    right-most vertices, and b and c as left and right vertices. 
    '''
    
    def __init__(self, name, minimum, b, c, maximum):
        LinguisticTerm.__init__(self, name, minimum, maximum)
        self.b = b 
        self.c = c
        
    def toFCL(self):
        return LinguisticTerm.toFCL(self) + \
        '(' + str(self.minimum) + ', ' + str(self.b) + ', ' + \
        str(self.c) + ', ' + str(self.maximum) + ')'

    def membership(self, crisp):
        if crisp < self.minimum or crisp > self.maximum: return 0.0
        
        mu = None
        if crisp < self.b:
            mu = Scale(crisp, 0.0, 1.0, self.minimum, self.b)
        elif crisp < self.c:
            mu = 1.0
        elif crisp < self.maximum:
            mu = Scale(crisp, 1.0, 0.0, self.c, self.maximum)
        else: return 0.0
         
        return self.fuzzy_operator.modulate(mu, self.modulation_degree);
        
class LeftShoulder(LinguisticTerm):
    '''Defines a left ramp term.'''
    def __init__(self, name, minimum, maximum):
        LinguisticTerm.__init__(self, name, minimum, maximum) 
    
    def toFCL(self):
        return LinguisticTerm.toFCL(self) + \
            '(' + str(self.minimum) + ', ' + str(self.maximum) + ')'
    
    def membership(self, crisp):
        mu = None
        if crisp < self.minimum: mu = 1.0
        elif crisp > self.maximum: mu = 0.0
        else: mu = Scale(crisp, 1.0, 0.0, self.minimum, self.maximum)
        return self.fuzzy_operator.modulate(mu, self.modulation_degree)
        
class RightShoulder(LinguisticTerm):
    '''Defines a right ramp term.'''
    def __init__(self, name, minimum, maximum):
        LinguisticTerm.__init__(self, name, minimum, maximum)
    
    def toFCL(self):
        return LinguisticTerm.toFCL(self) + \
            '(' + str(self.minimum) + ', ' + str(self.maximum) + ')'

    def membership(self, crisp):
        mu = None
        if crisp > self.maximum: mu = 1.0
        elif crisp < self.minimum: mu = 0.0
        else: mu = Scale(crisp, 0.0, 1.0, self.minimum, self.maximum)
        return self.fuzzy_operator.modulate(mu, self.modulation_degree)


class Singleton(Rectangular):
    '''Defines a singleton term that represents a single value bounded by epsilon.'''
    
    def __init__(self, name, value, epsilon = FuzzyOperator.Default().epsilon):
        Rectangular.__init__(self, name, value - epsilon, value + epsilon)
        self.value = value
        self.epsilon = epsilon
    
    def toFCL(self):
        return LinguisticTerm.toFCL(self) + \
            str(self.value) + '(' + str(self.epsilon) + ')'

class Function(LinguisticTerm):
    '''Defines a linguistic term by a function given as a lambda expression.'''
    def __init__(self, name, lambda_expression, minimum, maximum):
        LinguisticTerm.__init__(self, name, minimum, maximum)
        self.lambda_expression = lambda_expression
    
    def toFCL(self):
        return LinguisticTerm.toFCL(self) + str(self.lambda_expression) \
            + ', (' + str(self.minimum) + ', ' + str(self.maximum) + ')'
            
    def membership(self, crisp):
        return self.lambda_expression(crisp, self.minimum, self.maximum)

class Composite(LinguisticTerm):
    '''Defines a linguistic term with other terms.'''
    def __init__(self, name, terms = []):
        LinguisticTerm.__init__(self, name, float('-inf'), float('inf'))
        self.terms = terms;
    
    def toFCL(self):
        return LinguisticTerm.toFCL(self) + '{' +\
            ''.join('[' + term.toFCL() + ']' for term in self.terms) + '}'
    
    def membership(self, crisp):
        mu = 0.0
        for term in self.terms:
            mu = self.fuzzy_operator.aggregate(mu, term.membership(crisp))
        return self.fuzzy_operator.modulate(mu, self.modulation_degree) 

if __name__ == '__main__':
    
    
    #Test: Composite
    composite = Composite('mix')
    composite.terms.append(Triangular('a', 0, 5, 10))
    composite.terms.append(Rectangular('a', 0, 5))
    print(composite.toFCL())
    l = []
    l.append('pe')
    l.append('cue')
    l.append('ca')
    print(repr([x for x in list]))

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
