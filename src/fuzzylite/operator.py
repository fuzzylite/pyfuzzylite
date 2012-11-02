'''
Created on 10/10/2012

@author: jcrada
'''

from fuzzylite.integrator import Trapezoidal
from fuzzylite.defuzzifier import CenterOfGravity

def scale(x, to_min, to_max, from_min, from_max):
    '''Scales number x in range [from_min, from_max] to its equivalent in range [to_min, to_max].'''
    return (to_max - to_min) / (from_max - from_min) * (x - from_min) + to_min


class FuzzyAnd:
    @staticmethod
    def Min(a, b):
        '''Defines the AND operation as the minimum.''' 
        return min(a, b)
    
    @staticmethod
    def Prod(a, b):
        '''Defines the AND operation as the product.'''
        return a * b
    
    @staticmethod
    def BDiff(a, b):
        '''Defines the AND operation as the bounded difference.''' 
        return max(0, a + b - 1)


class FuzzyOr:
    @staticmethod
    def Max(a, b):
        '''Defines the OR operation as the maximum.'''
        return max(a, b)
    
    @staticmethod
    def Sum(a,b):
        '''Defines the OR operation as the algebraic sum.'''
        return a + b - (a * b)
    
    @staticmethod
    def BSum(a,b):
        '''Defines the OR operation as the algebraic bounded sum.'''
        return min(1, a + b)


class FuzzyModulate:
    
    @staticmethod
    def Clip(a, b):
        '''Defines the modulation as clipping.'''
        return min(a, b)
    
    @staticmethod
    def Scale(a,b):
        '''Defines the modulation as scaling.'''
        return a * b;

class Operator:
    '''
    Operator provides all the necessary operations to be performed.
    Multiple instances can be created, but also a default one is provided.
    '''
    instance = None
    @staticmethod
    def default():
        '''Retrieves the default fuzzy operator.'''
        if Operator.instance is None:
            Operator.instance = Operator()
        return Operator.instance
        
        
    def __init__(self, tnorm = FuzzyAnd.Min, snorm = FuzzyOr.Max, 
                 modulate = FuzzyModulate.Clip, aggregate = FuzzyOr.Max,
                 integrator = Trapezoidal, defuzzifier = CenterOfGravity,
                 sample_size = 100, epsilon = 1e-5):
        '''Constructs a Operator with default values.'''
        self.tnorm = tnorm
        self.snorm = snorm
        self.modulate = modulate
        self.aggregate = aggregate
        self.integrator = integrator
        self.defuzzifier = defuzzifier
        self.sample_size = sample_size
        self.epsilon = epsilon
    
    def defuzzify(self, term):
        return self.defuzzifier.defuzzify(term, self.integrator, self.sample_size)
    
    def area(self, term):
        return self.integrator.area(term, self.sample_size)
    
    def centroid(self, term):
        return self.integrator.centroid(term, self.sample_size)
    
    def area_and_centroid(self, term):
        return self.integrator.area_and_centroid(term, self.sample_size)
    



if __name__ == "__main__":
    print(Operator.default().tnorm)
#    Operator.default().tnorm = lambda a,b: a ** b
#    print(Operator.default().tnorm)
    fop = Operator.default()
    print('And: ', fop.tnorm(2,5))
    print('Or: ', fop.snorm(2,5))
    print('And: ', fop.tnorm)





















    