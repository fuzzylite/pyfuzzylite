'''
Created on 7/10/2012

@author: jcrada
'''
import logging


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


if __name__ == '__main__':
    x = FuzzyAnd.Min
    print(x(2, 5))
    x = FuzzyOr.Max
    print(x(2, 5))
    logging.error("Logging")
    pass
    
