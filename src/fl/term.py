"""
 pyfuzzylite (TM), a fuzzy logic control library in Python.
 Copyright (C) 2010-2017 FuzzyLite Limited. All rights reserved.
 Author: Juan Rada-Vilela, Ph.D. <jcrada@fuzzylite.com>

 This file is part of pyfuzzylite.

 pyfuzzylite is free software: you can redistribute it and/or modify it under
 the terms of the FuzzyLite License included with the software.

 You should have received a copy of the FuzzyLite License along with
 pyfuzzylite. If not, see <http://www.fuzzylite.com/license/>.

 pyfuzzylite is a trademark of FuzzyLite Limited
 fuzzylite is a registered trademark of FuzzyLite Limited.
"""

import abc
import math

import fl.exporter
from fl.engine import Engine
from fl.operation import Operation as Op

class Term(object):
    """
      The Term class is the abstract class for linguistic terms. The linguistic
      terms in this library can be divided in four groups as: `basic`,
      `extended`, `edge`, and `function`. The `basic` terms are Triangle,
      Trapezoid, Rectangle, and Discrete. The `extended` terms are Bell,
      Binary, Cosine, Gaussian, GaussianProduct, PiShape, SigmoidDifference,
      SigmoidProduct, and Spike. The `edge` terms are Concave, Ramp, Sigmoid,
      SShape, and ZShape. The `function` terms are Constant, Linear, and
      Function.

      In the figure below, the `basic` terms are represented in the first
      column, and the `extended` terms in the second and third columns. The
      `edge` terms are represented in the fifth and sixth rows, and the
      `function` terms in the last row.

      @image html terms.svg

      @author Juan Rada-Vilela, Ph.D.
      @see Variable
      @see InputVariable
      @see OutputVariable
      @since 4.0

      Attributes:
          name is the name of the term
          height is the height of the term
    """
    __metaclass__ = abc.ABCMeta
    __slots__ = 'name', 'height'

    def __init__(self, name="", height=1.0):
        self.name = name
        self.height = height

    def __str__(self):
        """
         Returns the representation of the term in the FuzzyLite Language
          :return the representation of the term in FuzzyLite Language
          @see FllExporter
        """
        return fl.exporter.FllExport().term(self)

    @abc.abstractmethod
    def parameters(self) -> str:
        """
          Returns the parameters to configure the term. The parameters are
          separated by spaces. If there is one additional parameter, the
          parameter will be considered as the height of the term; otherwise,
          the height will be set to @f$1.0@f$
          :return the parameters to configure the term (@see Term::configure())
         """
        return "" if self.height == 1.0 else Op.str(self.height)

    @abc.abstractmethod
    def configure(self, parameters) -> None:
        """
          Configures the term with the given parameters. The parameters are
          separated by spaces. If there is one additional parameter, the
          parameter will be considered as the height of the term; otherwise,
          the height will be set to @f$1.0@f$
          :param parameters is the parameters to configure the term
        """

    @abc.abstractmethod
    def membership(self, x) -> float:
        """
          Computes the membership function value at @f$x@f$
          :param x
          :return the membership function value @f$\mu(x)@f$
        """
        return None

    def update_reference(self, engine: Engine) -> None:
        """
          Updates the references (if any) to point to the current engine (useful
          when cloning engines or creating terms within Importer objects
          :param engine: is the engine to which this term belongs to
        """
        pass

    def tsukamoto(self, activation_degree: float, minimum: float, maximum: float) -> float:
        """
          For monotonic terms, computes the tsukamoto value of the term for the
          given activation degree @f$\alpha@f$, that is,
          @f$ g_j(\alpha) = \{ z \in\mathbb{R} : \mu_j(z) = \alpha \} $@f. If
          the term is not monotonic (or does not override this method) the
          method computes the membership function @f$\mu(\alpha)@f$.
          :param activationDegree: is the activationDegree
          :param minimum is the minimum value of the range of the term
          :param maximum is the maximum value of the range of the term
          :return the tsukamoto value of the term for the given activation degree
                  if the term is monotonic (or overrides this method), or
                  the membership function for the activation degree otherwise.
        """
        return self.membership(activation_degree)

    def is_monotonic(self):
        """
        Indicates whether the term is monotonic.
          :return whether the term is monotonic.
        """
        return False


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
        elif x < self.middle_vertex:
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
        else:
            return 0.0


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
        return 1.0 / (1 + math.exp(-self.a * (x - self.c)))


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
