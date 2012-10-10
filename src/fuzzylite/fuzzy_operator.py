'''
Created on 10/10/2012

@author: jcrada
'''

from fuzzylite.fuzzy_operation import FuzzyAnd, FuzzyOr, FuzzyModulate
from fuzzylite.integration import Trapezoidal
from fuzzylite.defuzzifier import CenterOfGravity


class FuzzyOperator:
    '''
    FuzzyOperator provides all the necessary operations to be performed.
    Multiple instances can be created, but also a default one is provided.
    '''
    instance = None
    @staticmethod
    def Default():
        '''Retrieves the default fuzzy operator.'''
        if FuzzyOperator.instance is None:
            FuzzyOperator.instance = FuzzyOperator()
        return FuzzyOperator.instance
        
        
    def __init__(self, tnorm = FuzzyAnd.Min, snorm = FuzzyOr.Max, 
                 modulate = FuzzyModulate.Clip, aggregate = FuzzyOr.Max,
                 integration = Trapezoidal, defuzzifier = CenterOfGravity,
                 sample_size = 100, epsilon = 1e-5):
        '''Constructs a FuzzyOperator with default values.'''
        self.tnorm = tnorm
        self.snorm = snorm
        self.modulate = modulate
        self.aggregate = aggregate
        self.integration = integration
        self.defuzzifier = defuzzifier
        self.sample_size = sample_size
        self.epsilon = epsilon
    
    def defuzzify(self, term):
        return self.defuzzifier.defuzzify(term, self.integration, self.sample_size)
    
    def area(self, term):
        return self.integration.area(term, self.sample_size)
    
    def centroid(self, term):
        return self.integration.centroid(term, self.sample_size)
    
    def area_and_centroid(self, term):
        return self.integration.area_and_centroid(term, self.sample_size)
    
if __name__ == "__main__":
    print(FuzzyOperator.Default().tnorm)
#    FuzzyOperator.Default().tnorm = lambda a,b: a ** b
#    print(FuzzyOperator.Default().tnorm)
    fop = FuzzyOperator.Default()
    print('And: ', fop.tnorm(2,5))
    print('Or: ', fop.snorm(2,5))
    print('And: ', fop.tnorm)
























    