'''
Created on 10/10/2012

@author: jcrada
'''

from math import sqrt

class HedgeSet:
    
    def __init__(self):
        self.hedge = {}
        self.hedge['not'] = lambda mu: 1.0 - mu
        self.hedge['somewhat'] = lambda mu: sqrt(mu)
        self.hedge['very'] = lambda mu: mu * mu
        self.hedge['any'] = lambda mu: 1.0


if __name__ == '__main__':
    hedge_set = HedgeSet()
    print(hedge_set.hedge['somewhat'](4))