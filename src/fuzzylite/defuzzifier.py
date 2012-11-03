'''
Created on 10/10/2012

@author: jcrada
'''

from fuzzylite.integrator import Rectangle

class Defuzzifier(object):
    
    def __init__(self, integrator = Rectangle(), samples = 100):
        self.integrator = integrator
        self.samples = 100
    
    def defuzzify(self, term):
        raise NotImplementedError('defuzzify')
    def toFCL(self):
        raise NotImplementedError('toFCL')
    
class CenterOfGravity(Defuzzifier):
    '''
    Defuzzifies a term according to the Center of Gravity
    '''
    def __init__(self, integrator = Rectangle(), samples = 100):
        Defuzzifier.__init__(integrator, samples)

    def defuzzify(self, term):
        return self.integrator.centroid(term, self.samples)
    
    def toFCL(self):
        return 'COG'
