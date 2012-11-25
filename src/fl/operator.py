'''
Created on 10/10/2012

@author: jcrada
'''


#To fulfill de Morgan's Law, the algorithms for operators AND and OR shall
#be used pair-wise e.g. MAX shall be used for OR if MIN is used for AND. [fcl, p.13]

class FuzzyAnd(object):
    @staticmethod
    def Min(a, b):
        '''Minimum.''' 
        return min(a, b)
    
    @staticmethod
    def Prod(a, b):
        '''Product.'''
        return a * b
    
    @staticmethod
    def BDif(a, b):
        '''Bounded difference.''' 
        return max(0, a + b - 1)


class FuzzyOr(object):
    @staticmethod
    def Max(a, b):
        '''Maximum.'''
        return max(a, b)
    
    @staticmethod
    def ASum(a, b):
        '''Algebraic sum.'''
        return a + b - (a * b)
    
    @staticmethod
    def BSum(a, b):
        '''Algebraic bounded sum.'''
        return min(1, a + b)


class FuzzyActivation(object):
    
    @staticmethod
    def Min(a, b):
        '''Defines the activation as clipping.'''
        return min(a, b)
    
    @staticmethod
    def Prod(a, b):
        '''Defines the activation as scaling.'''
        return a * b;
    
class FuzzyAccumulation(object):
    @staticmethod
    def Max(a, b):
        '''Maximum.'''
        return max(a, b)
    
    @staticmethod
    def BSum(a, b):
        '''Algebraic bounded sum.'''
        return min(1, a + b)
    
    @staticmethod
    def NSum(a, b):
        '''Normalized sum.'''
        return (a + b) / max(1, max(a, b))






