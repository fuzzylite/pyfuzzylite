'''
Created on 10/10/2012

@author: jcrada
'''

#TODO: Compute centroid in Trapezoidal ,Rectangle, and Simpson.

class Integrator(object):
    
    def area(self, term, sample_size):
        raise NotImplemented('area')
    
    def centroid(self, term, sample_size):
        raise NotImplemented('centroid')
    
    def area_and_centroid(self, term, sample_size):
        raise NotImplemented('area_and_centroid')
    
class Rectangle(Integrator):
    def area(self, term, sample_size):
        step_size = (term.maximum - term.minimum) / sample_size
        area = 0.0
        for i in range(1, sample_size + 1):
            area += term.membership(term.minimum + (i - 0.5) * step_size)
        return area * step_size
    
    def centroid(self, term, sample_size):
        raise NotImplemented('centroid')
    
    def area_and_centroid(self, term, sample_size):
        raise NotImplemented('area_and_centroid')

class Trapezoidal(Integrator):
    '''
    Integrates a term to compute its area and centroid using the trapezoidal rule 
    '''

    def area(self, term, sample_size):
        step_size = (term.maximum - term.minimum) / sample_size
        area = term.membership(term.minimum) + term.membership(term.maximum)
        for i in range(1, sample_size):
            area += 2.0 * term.membership(term.minimum + i * step_size)
        return area * step_size / 2.0

    def centroid(self, term, sample_size):
        (unused_area, centroid) = Trapezoidal.area_and_centroid(term, sample_size) 
        return centroid

    def area_and_centroid(self, term, sample_size):
        from math import isinf
        if isinf(term.maximum) or isinf(term.minimum):
            raise ValueError('cannot compute area on term that extends to infinity')
        sum_area = 0.0
        step_size = (term.maximum - term.minimum) / sample_size
        step = term.minimum
        mu = None
        previous_mu = term.membership(step);
        area = None; x = None; y = None;
        
        centroid_x = 0.0;
        centroid_y = 0.0;
        #
        #  centroid_x = a (h_1 + 2h_2)/3(h_1+h_2) ; h_1 = prev_mu; h_2 = mu
        #  centroid_y = (h_1^2/(h_1+h_2) + h_2) * 1/3
        #
        for unused_i in range(0, sample_size):
            step += step_size
            mu = term.membership(step)

            area = 0.5 * step_size * (previous_mu + mu);
            sum_area += area;

            x = (step_size * (previous_mu + 2.0 * mu) / (3.0 * (previous_mu + mu))
                    + (step - step_size))
            y = (previous_mu * previous_mu / (previous_mu + mu) + mu) / 3.0;

            centroid_x += area * x;
            centroid_y += area * y;

            previous_mu = mu;
        
        centroid_x /= sum_area;
        centroid_y /= sum_area;

        return (sum_area, (centroid_x, centroid_y))
        

class Simpson(Integrator):
    def area(self, term, sample_size):
        raise NotImplemented('area')
    
    def centroid(self, term, sample_size):
        raise NotImplemented('centroid')
    
    def area_and_centroid(self, term, sample_size):
        raise NotImplemented('area_and_centroid')

if __name__ == '__main__':
    from fuzzylite import term 
    term = term.Triangular('a',0,5,10)
    print(Trapezoidal().area(term, 2))
    print(Trapezoidal().area_and_centroid(term, 2))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        