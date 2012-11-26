'''
Created on 10/10/2012

@author: jcrada
'''


# TODO: Copy matlab membership functions
import math
import logging
class Term(object):
    '''Base class to define fuzzy linguistic terms such as LOW, MEDIUM, HIGH.
    
    Attributes:
            name: a string containing the name of the term (e.g. LOW, MEDIUM, HIGH)
            minimum: a float from which the term starts
            maximum: a float to which the term ends 
    '''
    
    def __init__(self, name, minimum, maximum):
        self.name = name
        self.minimum = minimum
        self.maximum = maximum
        self.logger = logging.getLogger(type(self).__name__)
        
    def __str__(self):
        '''Returns a string of this term.'''
        return '%s (%s, %s)' % (self.__class__.__name__, self.minimum, self.maximum)
    
    def discretize(self, divisions=100, align='center'):
        '''Discretizes terms.
        
        Terms are discretized using the resolution given by divisions.
        
        Args:
            divisions: the number of slices in which the term is divided.
            align: determines from alignment of the slice with respect to the step size.

        Returns:
            A generator with coordinates (x,y)
        
        Raises:
            ValueError: if the term is unbounded (i.e. tends to infinity)
        '''
        if  math.isinf(self.minimum) or math.isinf(self.maximum):
            raise ValueError('cannot discretize a term whose minimum or maximum is infinity')
        # dx is the step size
        dx = (self.maximum - self.minimum) / divisions
        x = None
        y = None
        shift = 0.0
        if align == 'left': pass
        elif align == 'center': shift = 0.5
        elif align == 'right': shift = 1.0
        else: raise ValueError('invalid align value <%s>' % align)
        for i in range(0, divisions):
            x = self.minimum + (i + shift) * dx
            y = self.membership(x)
            yield (x, y)
    
    def membership(self, x):
        '''Determines the degree of membership from crisp number x to the term.
        
        Args:
            x: a float number
        Returns:
            mu: the membership degree of x.
        Raises:
            NotImplementedError: if the term does not implement this method.''' 
        raise NotImplementedError()



class Triangle(Term):
    '''A triangular term.
    
    Defines a triangular term using minimum as the left vertex,
    middle as the center vertex, and maximum as the right vertex
    
    Attributes:
        minimum: the left vertex
        middle_vertex: the middle vertex
        maximum: the right vertex
    '''
    
    def __init__(self, name, minimum, middle_vertex, maximum):
        Term.__init__(self, name, minimum, maximum)
        self.middle_vertex = middle_vertex
    
    def __str__(self):
        return '%s (%s, %s, %s)' % (self.__class__.__name__,
                                    self.minimum, self.middle_vertex, self.maximum)

    def membership(self, x):
        if x <= self.minimum or x >= self.maximum:
            return 0.0
        elif x == self.middle_vertex:
            return 1.0
        elif x < self.middle_vertex :
            return (x - self.minimum) / (self.middle_vertex - self.minimum)
        else:
            return (self.maximum - x) / (self.maximum - self.middle_vertex) 

class Trapezoid(Term):
    '''A trapezoid term.
    
    Defines a trapezoidal term using minimum and maximum as left-most and
    right-most vertices, and b and c as left and right vertices.
    
    Attributes:
        minimum: leftmost vertex
        b: left vertex
        c: right vertex
        maximum: rightmost vertex
    '''
    
    def __init__(self, name, minimum, b, c, maximum):
        Term.__init__(self, name, minimum, maximum)
        self.b = b 
        self.c = c
        
    def __str__(self):
        return '%s (%s, %s, %s, %s)' % (self.__class__.__name__,
                                    self.minimum, self.b, self.c, self.maximum)
        

    def membership(self, x):
        if x <= self.minimum or x >= self.maximum: 
            return 0.0
        elif x <= self.b:
            return (x - self.minimum) / (self.b - self.minimum)
        elif x <= self.c:
            return 1.0
        elif x <= self.maximum:
            return (self.maximum - x) / (self.maximum - self.c)
        else: return 0.0
        
class Rectangle(Term):
    '''A rectangular term.
    
    Defines a rectangular term in the range [minimum, maximum].
    
    Attributes:
        minimum: left side
        maximum: right side
    '''
    def __init__(self, name, minimum, maximum):
        Term.__init__(self, name, minimum, maximum)
    
    def __str__(self):
        return '%s (%s, %s)' % (self.__class__.__name__,
                                    self.minimum, self.maximum)
    
    def membership(self, x):
        return 1.0 if self.minimum <= x <= self.maximum else 0.0
    
class LeftShoulder(Term):
    '''A left shoulder term. 
    mu=1.0 ___
              \
               \___ mu=0.0
    Defines a left shoulder term as shown above.'''
    
    def __init__(self, name, minimum, maximum):
        Term.__init__(self, name, minimum, maximum)
        
    
    def membership(self, x):
        if x <= self.minimum: return 1.0
        if x >= self.maximum: return 0.0
        return (x - self.minimum) / (self.maximum - self.minimum)
        

class RightShoulder(Term):
    '''A right shoulder term. 
                ___ mu=1.0
               /   
    mu=0.0 ___/      
    Defines a right shoulder term as shown above.'''
    
    def __init__(self, name, minimum, maximum):
        Term.__init__(self, name, minimum, maximum)
        
    
    def membership(self, x):
        if x <= self.minimum: return 0.0
        if x >= self.maximum: return 1.0
        return (self.maximum - x) / (self.maximum - self.minimum)


class Lambda(Term):
    '''A function term.
    
    Defines a linguistic term by a function given as a lambda expression.
    The lambda expression must contain handle three variables referent to the
    crisp value x, the minimum, and the maximum assigned to the term. 
    
    Attributes:
        lambda_expression: the function expressed as a lambda expression with  
                           parameters (x, minimum, maximum):
                           x: the float value.
                           minimum: the minimum value of the term.
                           maximum: the maximum value of the term.
    '''
    def __init__(self, name, strlambda, minimum, maximum):
        Term.__init__(self, name, minimum, maximum)
        self.strlambda = strlambda
        self.lambda_ = eval(strlambda)
    
    def __str__(self):
        return '%s %s (%s, %s)' % (self.__class__.__name__,
                                    self.strlambda,
                                    self.minimum, self.maximum)

    def membership(self, x):
        return self.lambda_(x, self.minimum, self.maximum)

class Gaussian(Term):
    '''Gaussian curve membership function

    from matlab docs:
    The symmetric Gaussian function depends on two parameters sigma and c as given 
    by
    
    f(x, sigma, c) = exp(-(x - c).^2/(2*sigma^2));
    
    '''
    def __init__(self, name, minimum, maximum, sigma, c):
        Term.__init__(self, minimum, maximum)
        self.sigma = sigma
        self.c = c
        
    def __str__(self):
        return '%s (%s, %s, %s, %s)' % (self.__class__.__name__,
                                        self.minimum, self.maximum,
                                        self.sigma, self.c)
    def membership(self, x):
        # from matlab: gaussmf.m
        return math.exp((-(x - self.c) ** 2) / (2 * self.sigma ** 2))
    
class Bell(Term):
    '''Generalized bell-shaped membership function
    
    from matlab docs:
    The generalized bell function depends on three parameters a, b, and c as given by:
        
        tmp = ((x - c)/a).^2;
        if (tmp == 0 & b == 0)
            y = 0.5;
        elseif (tmp == 0 & b < 0)
            y = 0;
        else
            tmp = tmp.^b;
            y = 1./(1 + tmp);
        end
    
    where the parameter b is usually positive. The parameter c locates the center
    of the curve. Enter the parameter vector params, the second argument for 
    gbellmf, as the vector whose entries are a, b, and c, respectively.
    '''
    
    def __init__(self, name, minimum, maximum, a, b, c):
        Term.__init__(self, minimum, maximum)
        self.a = a
        self.b = b
        self.c = c
        
    def __str__(self):
        return '%s (%s, %s, %s, %s, %s)' % (self.__class__.__name__,
                                        self.minimum, self.maximum,
                                        self.a, self.b, self.c)
        
    def membership(self, x):
        # from matlab: gbellmf.m
        tmp = ((x - self.c) / self.a) ** 2
        if tmp == 0.0 and self.b == 0:
            return 0.5
        elif tmp == 0.0 & self.b < 0:
            return 0.0
        else:
            tmp = tmp ** self.b
            return 1.0 / (1 + tmp)


class Sigmoid(Term):
    '''Sigmoidal membership function

    from matlab docs:
    The sigmoidal function, sigmf(x,[a c]), as given in the following equation 
    by f(x,a,c) is a mapping on a vector x, and depends on two parameters a and c.
    
    f(x,a,c) = 1./(1 + exp(-a*(x-c)));

    Depending on the sign of the parameter a, the sigmoidal membership function 
    is inherently open to the right or to the left, and thus is appropriate for 
    representing concepts such as 'very large' or 'very negative'.
    '''


    def __init__(self, name, minimum, maximum, a, c):
        Term.__init__(self, minimum, maximum)
        self.a = a
        self.c = c
    
    def __str__(self):
        return '%s (%s, %s, %s, %s)' % (self.__class__.__name__,
                                        self.minimum, self.maximum,
                                        self.a, self.c)
    
    def membership(self, x):
        # from matlab: sigmf.m 
        return 1.0 / (1 + math.exp( -self.a * (x - self.c)))


class Output(Term):
    '''An output term to be used in the Cumulative output term.
    
    Wraps any linguistic term to add the alphacut and an activation method.
     
    Attributes:
        term: a term  it wraps.
        alphacut: the float degree of the activation.
        activation: a method to define membership functions considering the alphacut.
                    It takes functions from FuzzyAnd'''
    def __init__(self, term, alphacut=1.0, activation=None):
        Term.__init__(self, term.name, term.minimum, term.maximum)
        self.term = term
        self.alphacut = alphacut
        self.activation = activation
    
    def __str__(self):
        return '%s %s %f' % (self.term, self.activation.__name__, self.alphacut)
    
    def membership(self, x):
        '''Returns the membership of x considering the alphacut'''
        if self.activation is None:
            raise ValueError('activation must take a FuzzyAnd function')
        return self.activation(self.term.membership(x), self.alphacut) 


class Cumulative(Term):
    '''A term made up with multiple terms.
    
    Defines a composite term made by others.
    
    Attributes:
        terms: a list of terms.
        accumulation: a FuzzyOr function that chooses the membership function 
            of overlapping terms.'''

    def __init__(self, name, accumulation=None):
        Term.__init__(self, name, float('-inf'), float('inf'))
        self.terms = []
        self.accumulation = accumulation
    
    def __str__(self):
        terms = ['[' + str(term) + ']' for term in self.terms]
        return '%s (%s, %s, %s)' % (self.__class__.__name__,
                                    self.minimum, self.maximum,
                                    ' , '.join(terms)) 
    
    def append(self, term):
        '''Appends a term to the list of terms.'''
        self.logger.debug('appending term: %s' % term)
        # the following updates the boundaries of this term.
        if math.isinf(self.minimum) or term.minimum < self.minimum:
            self.minimum = term.minimum
        if math.isinf(self.maximum) or term.maximum > self.maximum:
            self.maximum = term.maximum
        self.terms.append(term)
        
    def clear(self):
        '''Clears the term by removing all the terms it is made up with.'''
        self.logger.debug('clearing output')
        self.minimum = float('-inf')
        self.maximum = float('inf')
        self.terms = []
        
    def is_empty(self):
        '''Returns a boolean that indicates whether the term contains other terms.'''
        return len(self.terms) == 0
    
    def membership(self, x):
        '''Returns the membership of x using the accumulation function.'''
        if self.accumulation is None:
            raise ValueError('accumulation method cannot be None');
        mu = 0.0
        for term in self.terms:
            mu = self.accumulation(mu, term.membership(x))
        return mu 

if __name__ == '__main__':
    
    a = Triangle('Low', 0, 5, 10)
    for pair in a.discretize(10):
        print(pair)
    print(a)
    # Test: Composite
    composite = Cumulative('mix')
    composite.append(Triangle('a', 0, 5, 10))
    composite.append(Rectangle('a', 0, 5))
    print(composite)
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
