'''
Created on 10/10/2012

@author: jcrada
'''

class Defuzzifier(object):
    
    def __init__(self, divisions=100):
        self.divisions = divisions
    
    def __str__(self):
        return self.__class__.__name__
    
    def defuzzify(self, term):
        raise NotImplementedError('defuzzify')
    
    def toFCL(self):
        raise NotImplementedError('toFCL')
    
class CenterOfGravity(Defuzzifier):
    '''
    Defuzzifies a term according to the Center of Gravity
    '''
    def __init__(self, divisions=100):
        Defuzzifier.__init__(self, divisions)

    def defuzzify(self, term):
        xcentroid = ycentroid = 0.0
        area = 0.0
        for x, y in term.discretize(self.divisions):
            xcentroid += y * x
            ycentroid += y * y
            area += y
        xcentroid /= area
        ycentroid /= 2 * area
        dx = (term.maximum - term.minimum) / self.divisions
        area *= dx
        return (xcentroid, ycentroid)
    
    def toFCL(self):
        return 'COG'
    
class SmallestOfMaximum(Defuzzifier):
    '''Defuzzifies the a term according to the smallest of the maximum value'''
     
    def __init__(self, divisions=100):
        Defuzzifier.__init__(self, divisions)
    
    def defuzzify(self, term):
        xsmallest = None
        ymax = -1.0  
        for x, y in term.discretize(self.divisions):
            if y > ymax:
                xsmallest = x
                ymax = y
        return (xsmallest, ymax)
        
    def toFCL(self):
        return 'SOM'

class LargestOfMaximum(Defuzzifier):
    '''Defuzzifies the a term according to the largest of the maximum value'''
     
    def __init__(self, divisions=100):
        Defuzzifier.__init__(self, divisions)
    
    def defuzzify(self, term):
        xlargest = None
        ymax = -1.0  
        for x, y in term.discretize(self.divisions):
            if y >= ymax:
                xlargest = x
                ymax = y
        return (xlargest, ymax)
    
    def toFCL(self):
        return 'LOM'

class MiddleOfMaximum(Defuzzifier):
    '''Defuzzifies the a term according to the largest of the maximum value'''
     
    def __init__(self, divisions=100):
        Defuzzifier.__init__(self, divisions)
    
    def defuzzify(self, term):
        xsmallest = xlargest = None
        ymax = -1.0
        same_plateau = False  
        for x, y in term.discretize(self.divisions):
            if y > ymax:
                xsmallest = x
                ymax = y
                same_plateau = True
            elif y == ymax and same_plateau:
                xlargest = x
            elif y < ymax:
                same_plateau = False
                
        return ((xlargest + xsmallest) / 2.0, ymax)
    
    def toFCL(self):
        return 'MOM'
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
