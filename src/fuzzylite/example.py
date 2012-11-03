'''
Created on 30/10/2012

@author: jcrada
'''

from fuzzylite.engine import Engine
from fuzzylite.operator import Operator
from fuzzylite.variable import InputVariable, OutputVariable
from fuzzylite.term import Triangle
from fuzzylite.ruleblock import RuleBlock
from fuzzylite.mamdani import MamdaniRule
class Example(object):
    '''
    classdocs
    '''
    @staticmethod
    def simple_mamdani():
        fop = Operator.default()
        fe = Engine('simple-mamdani',fop=fop)
        energy = InputVariable('Energy')
        energy.terms['LOW'] = Triangle('LOW', 0.0, 0.25, 0.5)
        energy.terms['MEDIUM'] = Triangle('MEDIUM', 0.25, 0.5, 0.75)
        energy.terms['HIGH'] = Triangle('HIGH', 0.5, 0.75, 1.0)
        fe.input['Energy'] = energy
        
        health = OutputVariable('Health')
        health.terms['BAD'] = Triangle('BAD', 0.0, 0.25, 0.5)
        health.terms['REGULAR'] = Triangle('REGULAR', 0.25, 0.5, 0.75)
        health.terms['GOOD'] = Triangle('GOOD', 0.5, 0.75, 1.0)
        fe.output['Health'] = health
        
        rules = RuleBlock('')
        rules.append(MamdaniRule('if Energy is LOW then Health is BAD', fe))
        rules.append(MamdaniRule('if Energy is MEDIUM then Health is REGULAR', fe))
        rules.append(MamdaniRule('if Energy is HIGH then Health is GOOD', fe))
        fe.ruleblock[''] = rules
        
        input = 0.0
        while input <= 1.0:
            energy.input = input
            fe.process()
            print('Energy=%s\nEnergy is %s' % (energy.input, energy.fuzzify(input)))
            print('Output=%s\nHealth is %s' % (fop.defuzzify(health.output), health.fuzzify(input)))
            input += 0.1
            
            
        
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
    
    
    
    
    
    
    
    
    
    
    
