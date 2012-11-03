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
    
class Rectangle(Integrator):
    def area(self, term, samplesize):
        from math import isinf  
        if  isinf(term.minimum) or isinf(term.maximum):
            raise ValueError('cannot compute area on term that extends to infinity')
        
        stepsize = (term.maximum - term.minimum) / samplesize
        area = 0.0
        for i in range(1, samplesize + 1):
            area += term.membership(term.minimum + (i - 0.5) * stepsize)
        return area * stepsize
    
    def centroid(self, term, samplesize):
        (unused_area, centroid) = self.area_and_centroid(term, samplesize)
        return centroid
    
    
#    // use doubles if appropriate
#float xsum = 0.0;
#float ysum = 0.0;
#float area = 0.0;
#for(int i = 0; i < points.size - 1; i++) {
#    // I'm not a c++ guy... do you need to use pointers? You make the call here
#    Point p0 = points[i];
#    Point p1 = points[i+1];
#
#    double areaSum = (p0.x * p1.y) - (p1.x * p0.y)
#
#    xsum += (p0.x + p1.x) * areaSum;
#    ysum += (p0.y + p1.y) * areaSum;
#    area += areaSum;
#}
#
#float centMassX = xsum / (area * 6);
#float centMassY = ysum / (area * 6);
    
    def area_and_centroid(self, term, samplesize):
        from math import isinf  
        if  isinf(term.minimum) or isinf(term.maximum):
            raise ValueError('cannot compute area on term that extends to infinity')
        
        stepsize = (term.maximum - term.minimum) / samplesize
        area = 0.0
        centroid = [0.0, 0.0]
        for i in range(1, samplesize + 1):
            step = term.minimum + (i - 0.5) * stepsize
            mu = term.membership(step)
            area += mu
#            centroid[0] += mu * (2*step - stepsize)  
#            centroid[1] += stepsize * step  
        
        area *= stepsize
        centroid[0] /= area * samplesize
        centroid[1] /= area 
        return (area, centroid) 

class Trapezoid(Integrator):
    '''
    Integrates a term to compute its area and centroid using the trapezoidal rule
    '''

    def area(self, term, samplesize):
        stepsize = (term.maximum - term.minimum) / samplesize
        area = term.membership(term.minimum) + term.membership(term.maximum)
        for i in range(1, samplesize):
            area += 2.0 * term.membership(term.minimum + i * stepsize)
        return area * stepsize / 2.0

    def centroid(self, term, samplesize):
        (unused_area, centroid) = self.area_and_centroid(term, samplesize) 
        return centroid
    
    def area_and_centroid(self, term, samplesize):
        from math import isinf  
        if  isinf(term.minimum) or isinf(term.maximum):
            raise ValueError('cannot compute area on term that extends to infinity')
        stepsize = (term.maximum - term.minimum) / samplesize
        mu = term.membership(term.minimum)
        mu_next = None
        centroid = [0.0, 0.0]
        area = 0.0
        for i in range(1, samplesize):
            mu_next = term.membership(term.minimum + i * stepsize)
            area_i = mu + mu_next 
            area += area_i
            if mu + mu_next > 0: #convex polygon assumed
                centroid[0] += area_i * (stepsize * (mu + 2 * mu_next) / (3 * (mu + mu_next))
                                         + term.minimum + (i - 1) * stepsize)
                centroid[1] += area_i  *(1.0 / 3.0 * (mu_next + (mu * mu / (mu + mu_next))))
            mu = mu_next 
        
        area += mu + term.membership(term.maximum)
        
#            x = (stepsize * (previous_mu + 2.0 * mu) / (3.0 * (previous_mu + mu))
#                    + (step - stepsize))
#            y = (previous_mu * previous_mu / (previous_mu + mu) + mu) / 3.0;
        
        centroid[0] /= area
        centroid[1] /= area
        area *= stepsize / 2.0
        import logging
        logging.debug('testing area computation')
        if area != self.area(term, samplesize):
            raise RuntimeError('Buggy area')
        return (area, centroid)



if __name__ == '__main__':
    from fuzzylite import term
    terms = [
            term.Triangle('tri', 0, 0, 2),
             term.Rectangle('rect', 0, 10),
             term.Trapezoid('trap', 0, 2.5, 7.5, 10)]
    print('Areas:') 
    samplesize = 100
    for t in terms:
        print(str(t))
        print('R=%s' % Rectangle().area(t, samplesize))
        print('R=%s %s' % Rectangle().area_and_centroid(t, samplesize))
        print('T=%s' % Trapezoid().area(t, samplesize))
        print('T=%s %s' % Trapezoid().area_and_centroid(t, samplesize))
        print('---------------------------')
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
