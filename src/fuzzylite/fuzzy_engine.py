'''
Created on 10/10/2012

@author: jcrada
'''

from fuzzylite.fuzzy_operator import FuzzyOperator
from fuzzylite.hedge_set import HedgeSet

class FuzzyEngine:
    '''
    Wraps the whole system.
    '''


    def __init__(self, fuzzy_operator = FuzzyOperator.Default(),
                  hedge_set = HedgeSet()):
        self.fuzzy_operator = fuzzy_operator
        self.hedge_set = hedge_set
        
        