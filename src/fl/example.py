'''
Created on 30/10/2012

@author: jcrada
'''

from fl.engine import Engine, Operator
from fl.variable import InputVariable, OutputVariable
from fl.term import Triangle, LeftShoulder, RightShoulder
from fl.ruleblock import RuleBlock
from fl.mamdani import MamdaniRule

class Example(object):
    
    
        
    @staticmethod
    def simple_mamdani():
        fe = Engine('simple-mamdani')
        energy = InputVariable('Energy')
        energy.term['LOW'] = Triangle('LOW', 0.0, 0.5, 1.0)
        energy.term['MEDIUM'] = Triangle('MEDIUM', 0.5, 1.0, 1.5)
        energy.term['HIGH'] = Triangle('HIGH', 1.0, 1.5, 2.0)
        fe.input['Energy'] = energy
        
        health = OutputVariable('Health', default=float('nan'))
        health.term['BAD'] = LeftShoulder('BAD', 0.0, 0.5)
        health.term['REGULAR'] = Triangle('REGULAR', 0.5, 1.0, 1.5)
        health.term['GOOD'] = RightShoulder('GOOD', 1.0, 1.5)
        fe.output['Health'] = health
        
        ruleblock = RuleBlock()
        ruleblock.append(MamdaniRule.parse('if Energy is LOW then Health is BAD', fe))
        ruleblock.append(MamdaniRule.parse('if Energy is MEDIUM then Health is REGULAR', fe))
        ruleblock.append(MamdaniRule.parse('if Energy is HIGH then Health is GOOD', fe))
        fe.ruleblock[ruleblock.name] = ruleblock
        
        fe.configure(Operator())
        return fe
    
    @staticmethod
    def test_simple_mamdani():
        fe = Example.simple_mamdani()
        energy = fe.input['Energy']
        health = fe.output['Health']
        input_value = 0.0
        while input_value <= 2.0:
            energy.input = input_value
            fe.process()
            print(health.output)
            output = health.defuzzify()
            print('Energy=%f' % energy.input)
            print('Energy is %s' % energy.fuzzify(input_value))
            print('Output=%f' % output)
            print('Health is %s'% health.fuzzify(output)) 
            print('-------')
            input_value += 0.1
        

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
    Example.test_simple_mamdani()
    
    
    
    
    
    
    
    
    
    
    
    
    
