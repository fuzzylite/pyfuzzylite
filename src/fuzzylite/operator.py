'''
Created on 10/10/2012

@author: jcrada
'''




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
    def Sum(a, b):
        '''Defines the OR operation as the algebraic sum.'''
        return a + b - (a * b)
    
    @staticmethod
    def BSum(a, b):
        '''Defines the OR operation as the algebraic bounded sum.'''
        return min(1, a + b)


class FuzzyModulate:
    
    @staticmethod
    def Clip(a, b):
        '''Defines the modulation as clipping.'''
        return min(a, b)
    
    @staticmethod
    def Scale(a, b):
        '''Defines the modulation as scaling.'''
        return a * b;

from fuzzylite.defuzzifier import CenterOfGravity
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
            Operator.instance = Operator('Default')
        return Operator.instance
        
        
    def __init__(self, name, tnorm=FuzzyAnd.Min, snorm=FuzzyOr.Max,
                 activation=FuzzyModulate.Clip, accumulation=FuzzyOr.Max,
                 defuzzifier=CenterOfGravity()):
        '''Constructs a Operator with default values.'''
        self.name = name
        self.tnorm = tnorm
        self.snorm = snorm
        self.activation = activation
        self.accumulation = accumulation
        self.defuzzifier = defuzzifier
    
    def __str__(self):
        result = ['Operator %s' % self.name]
        result.append('tnorm = %s' % self.tnorm.__name__)
        result.append('snorm = %s' % self.snorm.__name__)
        result.append('activation = %s' % self.activation.__name__)
        result.append('accumulation = %s' % self.accumulation.__name__)
        result.append('defuzzifier = %s' % self.defuzzifier)
        return '\n'.join(result)
    
    def toFCL(self):
        from fuzzylite.rule import Rule
        fcl = []
        fcl.append('%s : %s;' % (Rule.FR_AND.upper(),
                                       self.tnorm.__name__.upper()))
        fcl.append('%s : %s;' % (Rule.FR_OR.upper(),
                                       self.snorm.__name__.upper()))
        fcl.append('ACT : %s;' % self.activation.__name__.upper())
        fcl.append('ACCU : %s;' % self.accumulation.__name__.upper())
        fcl.append('METHOD : %s;' % self.defuzzifier.toFCL())
        return '\n'.join(fcl)


if __name__ == "__main__":
    print(Operator.default().tnorm)
#    Operator.default().tnorm = lambda a,b: a ** b
#    print(Operator.default().tnorm)
    fop = Operator.default()
    print('And: ', fop.tnorm(2, 5))
    print('Or: ', fop.snorm(2, 5))
    print('And: ', fop.tnorm)
    print(fop)





















    
