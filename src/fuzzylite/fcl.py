'''
Created on 21/11/2012

@author: jcrada
'''
from fuzzylite.engine import Engine

class FCLExporter(object):
    '''Imports and exports fuzzy engines from and to the Fuzzy Controller Language'''

    def __init__(self):
        pass 
    
    def engine(self, fe):
        fcl = ['FUNCTION_BLOCK %s' % fe.name]
        fcl.append('')
        
        fcl.append('VAR_INPUT')
        for name in fe.input:
            fcl.append('%s: REAL;' % name)
        fcl.append('END_VAR')
        fcl.append('')
        
        fcl.append('VAR_OUTPUT')
        for name in fe.output:
            fcl.append('%s: REAL;' % name)
        fcl.append('END_VAR')
        fcl.append('')
        
        
        for name, variable in fe.input.items():
            fcl.append('FUZZIFY %s' % name)
            for term in variable:
                fcl.append('TERM %s := %s;' % (term.name, term))
            fcl.append('END_FUZZIFY')
            fcl.append('')
        
        for name, variable in fe.output.items():
            fcl.append('DEFUZZIFY %s' % name)
            for term in variable:
                fcl.append('TERM %s := %s;' % (term.name, term))
            fcl.append('')
            fcl.append('METHOD : %s;' % variable.defuzzifier)
            fcl.append('ACCU : %s;' % variable.output.accumulation.__name__.upper())
            if variable.default is not None:
                fcl.append('DEFAULT : %f;' % variable.default)
            fcl.append('END_FUZZIFY')
            fcl.append('')    

        from fuzzylite.rule import Rule
        for name, ruleblock in fe.ruleblock.items():
            fcl.append('RULEBLOCK %s' % name)
            fcl.append('%s : %s;' % (Rule.FR_AND.upper(),
                                       ruleblock.fop.tnorm.__name__.upper()))
            fcl.append('%s : %s;' % (Rule.FR_OR.upper(),
                                       ruleblock.fop.snorm.__name__.upper()))
            fcl.append('ACT : %s;' % ruleblock.fop.activation.__name__.upper())
            fcl.append('')
            for i, rule in enumerate(ruleblock):
                fcl.append('RULE %i : %s;' % (i + 1, rule))
            
            fcl.append('END_RULEBLOCK')
            fcl.append('')

        fcl.append('END_FUNCTION_BLOCK')
        return '\n'.join(fcl)

class FCLImporter(object):
    
    def __init__(self):
        pass
    
    def engine(self, fcl):
        fe = Engine()
        return fe

if __name__ == '__main__':
    from fuzzylite.example import Example
    fe = Example().simple_mamdani()
    export = FCLExporter()
    print(export.engine(fe))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    