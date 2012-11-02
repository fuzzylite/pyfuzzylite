'''
Created on 10/10/2012

@author: jcrada
'''

from fuzzylite.operator import Operator


#TODO: Copy matlab membership functions
class Term:
    '''
    Provides the necessary properties for any linguistic term
    '''
    
    def __init__(self, name, minimum, maximum, alphacut=1.0,
                 fop=Operator.default()):
        self.name = name
        self.minimum = minimum
        self.maximum = maximum
        self.alphacut = alphacut
        self.fop = fop
        
    def toFCL(self):
        '''Returns a string in Fuzzy Control Language.'''
        return 'TERM ' + str(self.name) + ' := ' + str(self)
    
    def membership(self, x): raise NotImplementedError()


class Triangular(Term):
    '''
    Defines a triangular term using minimum as the left vertex,
    middle as the center vertex, and maximum as the right vertex
    '''
    
    def __init__(self, name, minimum, middle_vertex, maximum):
        Term.__init__(self, name, minimum, maximum)
        self.middle_vertex = middle_vertex
    
    def __str__(self):
        return str(self.__class__.__name__) + \
        '(' + str(self.minimum) + ', ' + \
        str(self.middle_vertex) + ', ' + str(self.maximum) + ')'

    def membership(self, x):
        mu = None
        if x <= self.minimum or x >= self.maximum:
            mu = 0.0
        elif x == self.middle_vertex:
            mu = 1.0
        elif x < self.middle_vertex :
            mu = (x - self.minimum) / (self.middle_vertex - self.minimum)
        else:
            mu = (self.maximum - x) / (self.maximum - self.middle_vertex) 
        return self.fop.modulate(mu, self.alphacut);

class Trapezoidal(Term):
    '''
    Defines a trapezoidal term using minimum and maximum as left-most and
    right-most vertices, and b and c as left and right vertices.
    '''
    
    def __init__(self, name, minimum, b, c, maximum):
        Term.__init__(self, name, minimum, maximum)
        self.b = b 
        self.c = c
        
    def __str__(self):
        return str(self.__class__.__name__) + \
        '(' + str(self.minimum) + ', ' + str(self.b) + ', ' + \
        str(self.c) + ', ' + str(self.maximum) + ')'

    def membership(self, x):
        mu = None
        if x <= self.minimum or x >= self.maximum: 
            mu = 0.0
        elif x <= self.b:
            mu = (x - self.minimum) / (self.b - self.minimum)
        elif x <= self.c:
            mu = 1.0
        elif x <= self.maximum:
            mu = (self.maximum - x) / (self.maximum - self.c)
        else: mu = 0.0

        return self.fop.modulate(mu, self.alphacut);
        
class Rectangular(Term):
    '''Defines a rectangular term in the range [minimum, maximum].'''
    def __init__(self, name, minimum, maximum):
        Term.__init__(self, name, minimum, maximum)
    
    def __str__(self):
        return str(self.__class__.__name__) + \
            '(' + str(self.minimum) + ', ' + str(self.maximum) + ')'
    
    def membership(self, x):
        mu = None
        if x < self.minimum or x > self.maximum: 
            mu = 0.0 
        else: mu = 1.0
        return self.fop.modulate(mu, self.alphacut)

class Function(Term):
    '''Defines a linguistic term by a function given as a lambda expression.'''
    def __init__(self, name, lambda_expression, minimum, maximum):
        Term.__init__(self, name, minimum, maximum)
        self.lambda_expression = lambda_expression
    
    def __str__(self):
        return str(self.__class__.__name__) + str(self.lambda_expression) + \
            ', (' + str(self.minimum) + ', ' + str(self.maximum) + ')'
            
    def membership(self, x):
        return self.lambda_expression(x, self.minimum, self.maximum)

class Composite(Term):
    '''Defines a linguistic term with other terms.'''
    def __init__(self, name):
        Term.__init__(self, name, float('-inf'), float('inf'))
        self.terms = [];
    
    def __str__(self):
        return str(self.__class__.__name__) + '{' + \
            ' , '.join('[' + str(term) + ']' for term in self.terms) + '}'
    
    def aggregate(self, term):
        import math
        if math.isinf(self.minimum) or term.minimum < self.minimum:
            self.minimum = term.minimum
        if math.isinf(self.maximum) or term.maximum > self.maximum:
            self.maximum = term.maximum
        self.terms.append(term)
        
    def clear(self):
        self.minimum = float('-inf')
        self.maximum = float('inf')
        self.terms = []
    
    def toFCL(self):
        return 'TERM ' + str(self.name) + ' := ' + \
            str(self.__class__.__name__) + '{' + \
            ' , '.join('[' + term.toFCL() + ']' for term in self.terms) + '}'
    
    def membership(self, x):
        mu = 0.0
        for term in self.terms:
            mu = self.fop.aggregate(mu, term.membership(x))
        return self.fop.modulate(mu, self.alphacut) 

if __name__ == '__main__':
    
    a = Triangular('Low', 0, 5, 10)
    print(a)
    print(a.toFCL())
    #Test: Composite
    composite = Composite('mix')
    composite.terms.append(Triangular('a', 0, 5, 10))
    composite.terms.append(Rectangular('a', 0, 5))
    print(composite)
    print(composite.toFCL())
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
