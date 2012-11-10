'''
Created on 10/10/2012

@author: jcrada
'''

# TODO: Compute centroid in Trapezoid ,Rectangle, and Simpson.

class Integrator(object):
    
    def area(self, term, samplesize):
        raise NotImplementedError('area')
    
    def centroid(self, term, samplesize):
        raise NotImplementedError('centroid')
    
    def area_and_centroid(self, term, samplesize):
        raise NotImplementedError('area_and_centroid')
    
from math import isinf

class Rectangle(Integrator):  # ready
    def area(self, term, samplesize):
        if  isinf(term.minimum) or isinf(term.maximum):
            raise ValueError('cannot compute area on term that extends to infinity')
        dx = (term.maximum - term.minimum) / samplesize
        area = 0.0
        for i in range(0, samplesize):
            area += term.membership(term.minimum + (i + 0.5) * dx)
        return area * dx
    
    def aread(self, term, divisions):
        area = 0.0
        for unused_x, y in term.discretize(divisions=divisions, align='center'):
            area += y
        dx = (term.maximum - term.minimum) / (divisions)
        return area * dx
    
    def centroid(self, term, samplesize):
        (unused_area, centroid) = self.area_and_centroid(term, samplesize)
        return centroid
    
    def area_and_centroidd(self, term, divisions):
        xcentroid = ycentroid = 0.0
        area = 0.0
        for x, y in term.discretize(divisions,align='center'):
            xcentroid += y * x
            ycentroid += y * y
            area += y
        xcentroid /= area
        ycentroid /= 2 * area
        dx = (term.maximum - term.minimum) / divisions
        area *= dx
        return (area, [xcentroid, ycentroid])
     
    def area_and_centroid(self, term, samplesize):
        if  isinf(term.minimum) or isinf(term.maximum):
            raise ValueError('cannot compute area on term that extends to infinity')
        
        dx = (term.maximum - term.minimum) / samplesize
        area = 0.0
        xcentroid = ycentroid = 0.0
        for i in range(0, samplesize):
            x = term.minimum + (i + 0.5) * dx
            y = term.membership(x)
            xcentroid += y * x
            ycentroid += y * y
            area += y
        
        xcentroid /= area
        ycentroid /= 2 * area 
        area *= dx
        
        return (area, [xcentroid, ycentroid]) 





if __name__ == '__main__':
    from fuzzylite import term
    terms = [
            term.Triangle('tri', 0, 0, 1), term.Triangle('tri', 0, 0.5, 1),
            term.Triangle('tri', 0, 1, 1),
             term.Rectangle('rect', 0, 10),
             term.Trapezoid('trap', 0, 0.25, 0.75, 1)]
    print('Areas:') 
    samplesize = 100
    for t in terms:
        print(str(t))
        print('R=%s' % Rectangle().area(t, samplesize))
        print('R=%s' % Rectangle().aread(t, samplesize))
        print('R=%s %s' % Rectangle().area_and_centroid(t, samplesize))
        print('R=%s %s' % Rectangle().area_and_centroidd(t, samplesize))
#        print('T=%s' % Trapezoid().area(t, samplesize))
#        print('T=%s' % Trapezoid().aread(t, samplesize))
#        print('R=%s %s' % Trapezoid().area_and_centroid(t, samplesize))
#        print('R=%s %s' % Trapezoid().area_and_centroidd(t, samplesize))
        print('---------------------------')
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
