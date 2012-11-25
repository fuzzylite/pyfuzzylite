'''
Created on 10/10/2012

@author: jcrada
'''
import logging

class Defuzzifier(object):
    
    def __init__(self, divisions=100):
        self.divisions = divisions
        self._logger = logging.getLogger(__name__) 
    
    def __str__(self):
        acronym = []
        for letter in self.__class__.__name__:
            if letter.isupper(): acronym.append(letter)
        return ''.join(acronym)
    
    def defuzzify(self, term):
        raise NotImplementedError('defuzzify')
    
class CenterOfGravity(Defuzzifier):
    '''
    Defuzzifies a term according to the Center of Gravity
    '''
    def __init__(self, divisions=100):
        Defuzzifier.__init__(self, divisions)

    def defuzzify(self, term):
        '''Defuzzifies the term by computing the centroid of the term'''
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
        
        self._logger.debug('centroid at (%f, %f)' % (xcentroid, ycentroid))
        return xcentroid
    
    
    
class SmallestOfMaximum(Defuzzifier):
    '''Defuzzifies the a term according to the smallest of the maximum value'''
     
    def __init__(self, divisions=100):
        Defuzzifier.__init__(self, divisions)
    
    def defuzzify(self, term):
        '''Defuzzifies the term by locating the leftmost x of the maximum membership function'''
        xsmallest = None
        ymax = -1.0  
        for x, y in term.discretize(self.divisions):
            if y > ymax:
                xsmallest = x
                ymax = y
        
        self._logger.debug('centroid at (%f, %f)' % (xsmallest, ymax))
        return xsmallest

class LargestOfMaximum(Defuzzifier):
    '''Defuzzifies the a term according to the largest of the maximum value'''
     
    def __init__(self, divisions=100):
        Defuzzifier.__init__(self, divisions)
    
    def defuzzify(self, term):
        '''Defuzzifies the term by locating the rightmost x of the maximum membership function'''
        xlargest = None
        ymax = -1.0  
        for x, y in term.discretize(self.divisions):
            if y >= ymax:
                xlargest = x
                ymax = y
        
        self._logger.debug('centroid at (%f, %f)' % (xlargest,ymax))
        return xlargest

class MiddleOfMaximum(Defuzzifier):
    '''Defuzzifies the a term according to the middle of the maximum value'''
     
    def __init__(self, divisions=100):
        Defuzzifier.__init__(self, divisions)
    
    def defuzzify(self, term):
        '''Defuzzifies the term by first locating the smallest and largest x of the maximum
        membership function, and then locating the middle by a simple average'''
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
        
        self._logger.debug('centroid at (%f, %f)' % ((xlargest + xsmallest) / 2.0, ymax))
        return (xlargest + xsmallest) / 2.0
#        return ((xlargest + xsmallest) / 2.0, ymax)

if __name__ == '__main__':
    x = MiddleOfMaximum()
    print(x)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
