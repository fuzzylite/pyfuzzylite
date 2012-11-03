'''
Created on 10/10/2012

@author: jcrada
'''
#from fuzzylite.fuzzy_operator import FuzzyOperator

class Defuzzifier(object):
    def defuzzify(self, term, integrator, sample_size):
        raise NotImplementedError('defuzzify')
    
class CenterOfGravity(Defuzzifier):
    '''
    Defuzzifies a term according to the Center of Gravity
    '''

    def defuzzify(self, term, integrator, sample_size):
        return integrator.centroid(term, sample_size)
    
