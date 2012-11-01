'''
Created on 30/10/2012

@author: jcrada
'''

from fuzzylite.fuzzy_engine import FuzzyEngine
from fuzzylite.variable import InputVariable, OutputVariable
from fuzzylite.membership_function import Triangular
class Example(object):
    '''
    classdocs
    '''
    @staticmethod
    def simple_mamdani():
        fe = FuzzyEngine('simple-mamdani')
        energy = InputVariable('Energy')
        energy.terms['LOW'] = Triangular('LOW', 0.0, 0.25, 0.5)
        energy.terms['MEDIUM'] = Triangular('MEDIUM', 0.25, 0.5, 0.75)
        energy.terms['HIGH'] = Triangular('HIGH', 0.5, 0.75, 1.0)
        fe.input['Energy'] = energy
        
        health = OutputVariable('Health')
        health.terms['BAD'] = Triangular('BAD', 0.0, 0.25, 0.5)
        health.terms['REGULAR'] = Triangular('REGULAR', 0.25, 0.5, 0.75)
        health.terms['GOOD'] = Triangular('GOOD', 0.5, 0.75, 1.0)
        fe.output['Health'] = health
        
        return fe
        

#        fl::RuleBlock * block = new fl::RuleBlock();
#        block -> addRule(new fl::MamdaniRule("if Energy is LOW then Health is BAD", engine));
#        block -> addRule(new fl::MamdaniRule("if Energy is MEDIUM then Health is REGULAR", engine));
#        block -> addRule(new fl::MamdaniRule("if Energy is HIGH then Health is GOOD", engine));
#        engine.addRuleBlock(block);
#
#        for (fl::flScalar in=0.0; in < 1.1; in +=0.1) {
#            energy -> setInput(in);
#            engine.process();
#            fl::flScalar out = health -> output().defuzzify();
#            (void)out; // Just to avoid warning when building
#            FL_LOG("Energy=" << in);
#            FL_LOG("Energy is " << energy -> fuzzify(in));
#            FL_LOG("Health=" << out);
#            FL_LOG("Health is " << health -> fuzzify(out));
#            FL_LOG("--");
#        }


if __name__ == '__main__':
    fe = Example.simple_mamdani()
    print (fe.toFCL())
    
    
    
    
    
    
    
    
    
    
    
