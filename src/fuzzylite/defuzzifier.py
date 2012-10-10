'''
Created on 10/10/2012

@author: jcrada
'''
#from fuzzylite.fuzzy_operator import FuzzyOperator


class CenterOfGravity:
    '''
    Defuzzifies a term according to the Center of Gravity
    '''

    @staticmethod
    def defuzzify(term, integration, sample_size ):
        return integration.centroid(term, sample_size)
    
    
        
        