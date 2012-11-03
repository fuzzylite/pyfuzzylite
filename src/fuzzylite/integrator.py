'''
Created on 10/10/2012

@author: jcrada
'''

#TODO: Compute centroid in Trapezoid ,Rectangle, and Simpson.

class Integrator(object):
    
    def area(self, term, samplesize):
        raise NotImplementedError('area')
    
    def centroid(self, term, samplesize):
        raise NotImplementedError('centroid')
    
    def area_and_centroid(self, term, samplesize):
        raise NotImplementedError('area_and_centroid')
    
from math import isinf

class Rectangle(Integrator): #ready
    def area(self, term, samplesize):
        if  isinf(term.minimum) or isinf(term.maximum):
            raise ValueError('cannot compute area on term that extends to infinity')
        dx = (term.maximum - term.minimum) / samplesize
        area = 0.0
        for i in range(0, samplesize):
            area += term.membership(term.minimum + (i + 0.5) * dx)
        return area * dx
    
    def centroid(self, term, samplesize):
        (unused_area, centroid) = self.area_and_centroid(term, samplesize)
        return centroid
    
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

import logging
class Trapezoid(Integrator):
    '''
    Integrates a term to compute its area and centroid using the trapezoidal rule.
    The results are not as precised as those from Rectangle, which could be from
    a very hard-to-find bug.
    '''

    def area(self, term, samplesize):
        logging.warn('fuzzylite.Trapezoid() integration is buggy')
        if  isinf(term.minimum) or isinf(term.maximum):
            raise ValueError('cannot compute area on term that extends to infinity')
        dx = (term.maximum - term.minimum) / samplesize
        area = term.membership(term.minimum) + term.membership(term.maximum)
        for i in range(1, samplesize):
            area += 2.0 * term.membership(term.minimum + i * dx)
        area *= 0.5 * dx
        return area 

    def centroid(self, term, samplesize):
        (unused_area, centroid) = self.area_and_centroid(term, samplesize) 
        return centroid
    
    def area_and_centroid(self, term, samplesize):
        logging.warn('fuzzylite.Trapezoid() integration is buggy')
        if  isinf(term.minimum) or isinf(term.maximum):
            raise ValueError('cannot compute area on term that extends to infinity')
        dx = (term.maximum - term.minimum) / samplesize
        y0 = term.membership(term.minimum)
        y1 = None
        xcentroid = ycentroid = 0.0
        area = 0
        for i in range(1, samplesize+1):
            y1 = term.membership(term.minimum + i * dx)
            area_i = y0 + y1 
            area += area_i
            if y0 + y1 > 0: #convex polygon assumed
                xcentroid += area_i * (dx * (y0 + 2 * y1) / (3 * (y0 + y1))
                                         + term.minimum + (i - 1) * dx)
                ycentroid += area_i * (1.0 / 3.0 * (y1 + (y0 * y0 / (y0 + y1))))
            y0 = y1 
        
        area += y0 + term.membership(term.maximum)
        
        xcentroid /= area
        ycentroid /= area
        area *= 0.5 * dx
        
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
        print('R=%s %s' % Rectangle().area_and_centroid(t, samplesize))
        print('T=%s' % Trapezoid().area(t, samplesize))
        print('T=%s %s' % Trapezoid().area_and_centroid(t, samplesize))
        print('---------------------------')
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
