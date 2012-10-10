'''
Created on 10/10/2012

@author: jcrada
'''



class Trapezoidal:
    '''
    Integrates a term to compute its area and centroid using the trapezoidal rule 
    '''


    @staticmethod
    def area(term, sample_size):
        sum_area = 0.0
        step_size = (term.maximum - term.minimum) / sample_size
        step = term.minimum
        mu = None
        previous_mu = term.membership(step)
        
        for i in range(0, sample_size):
            step += step_size
            mu = term.membership(step)
            
            sum_area += step_size * (mu + previous_mu)
            previous_mu = mu
        
        sum_area *= 0.5
        return sum_area

    @staticmethod
    def centroid(term, sample_size):
        (area, centroid) = Trapezoidal.area_and_centroid(term, sample_size) 
        return centroid

    @staticmethod
    def area_and_centroid(term, sample_size):
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
        for i in range(0, sample_size):
            step += step_size
            mu = term.membership(step)

            area = 0.5 * step_size * (previous_mu + mu);
            sum_area += area;

            x = ((step_size * (previous_mu + 2 * mu)) / (3.0 * (previous_mu + mu)))\
                    + (step - step_size)
            y = ((previous_mu * previous_mu) / (previous_mu + mu) + mu) / 3.0;

            centroid_x += area * x;
            centroid_y += area * y;

            previous_mu = mu;
        
        centroid_x /= sum_area;
        centroid_y /= sum_area;

        return (sum_area, (centroid_x, centroid_y))
        
        
        

if __name__ == '__main__':
    from fuzzylite.linguistic_term import Triangular
    term = Triangular('a',0,5,10)
    print(Trapezoidal.area_and_centroid(term, 100))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        