'''
Created on 10/10/2012

@author: jcrada
'''

import math

class Hedge(object):
    def __init__(self, name, function=None):
        self.name = name
        self.function = function
    
    def apply(self, mu): 
        return self.function(mu) 

class HedgeDict(dict):
    
    def __init__(self):
        dict.__init__(self)
        self['not'] = Hedge('not', function=lambda mu: 1.0 - mu)
        self['somewhat'] = Hedge('somewhat', function=lambda mu: math.sqrt(mu))
        self['very'] = Hedge('very', function=lambda mu: mu * mu)
        self['any'] = Hedge('any', function=lambda mu: 1.0)


if __name__ == '__main__':
    hedges = HedgeDict()
    print(hedges['any'].apply(4))
