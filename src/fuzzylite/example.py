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

import logging

class Example(object):
    '''
    classdocs
    '''
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        

    def simple_mamdani(self):
        fe = Engine('simple-mamdani')
        energy = InputVariable('Energy')
        energy.term['LOW'] = Triangle('LOW', 0.0, 0.5, 1.0)
        energy.term['MEDIUM'] = Triangle('MEDIUM', 0.5, 1.0, 1.5)
        energy.term['HIGH'] = Triangle('HIGH', 1.0, 1.5, 2.0)
        fe.input['Energy'] = energy
        
        health = OutputVariable('Health', default=float('nan'))
        health.term['BAD'] = Triangle('BAD', 0.0, 0.5, 1.0)
        health.term['REGULAR'] = Triangle('REGULAR', 0.5, 1.0, 1.5)
        health.term['GOOD'] = Triangle('GOOD', 1.0, 1.5, 2.0)
        fe.output['Health'] = health
        
        rules = RuleBlock('')
        rules.append(MamdaniRule('if Energy is LOW then Health is BAD', fe))
        rules.append(MamdaniRule('if Energy is MEDIUM then Health is REGULAR', fe))
        rules.append(MamdaniRule('if Energy is HIGH then Health is GOOD', fe))
        fe.ruleblock[''] = rules
        
        fe.configure(Operator.default())
        input = 0.0
        while input <= 2.0:
            energy.input = input
            fe.process()
            self._logger.info(health.output)
            output = health.defuzzify()
            self._logger.info('Energy=%f', energy.input)
            self._logger.info('Energy is %s', energy.fuzzify(input))
            self._logger.info('Output=%f' , output)
            self._logger.info('Health is %s', health.fuzzify(output)) 
            self._logger.info('-------')
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
    import logging
    logging.basicConfig(level = logging.DEBUG,
                        format = '%(asctime)s - %(name)-20s - %(levelname)s - %(message)s')
    fe = Example().simple_mamdani()
    
    
    
    
    
    
    
    
    
    
    
    
