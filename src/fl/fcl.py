'''
Created on 21/11/2012

@author: jcrada
'''
from fl.engine import Engine

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
            
            accu = variable.output.accumulation
            if accu is not None: accu = accu.__name__.upper()
            fcl.append('ACCU : %s;' % accu)
            
            if variable.default is not None:
                fcl.append('DEFAULT : %f;' % variable.default)
            fcl.append('END_DEFUZZIFY')
            fcl.append('')    

        from fl.rule import Rule
        for name, ruleblock in fe.ruleblock.items():
            fcl.append('RULEBLOCK %s' % name)
            
            tnorm = ruleblock.tnorm
            if tnorm is not None: tnorm = tnorm.__name__.upper()
            fcl.append('%s : %s;' % (Rule.FR_AND.upper(), tnorm))
            
            snorm = ruleblock.snorm
            if snorm is not None: snorm = snorm.__name__.upper()
            fcl.append('%s : %s;' % (Rule.FR_OR.upper(), snorm))
            
            activation = ruleblock.activation
            if activation is not None: activation = activation.__name__.upper()
            fcl.append('ACT : %s;' % activation)
            
            fcl.append('')
            for i, rule in enumerate(ruleblock):
                fcl.append('RULE %i : %s;' % (i + 1, rule))
            
            fcl.append('END_RULEBLOCK')
            fcl.append('')

        fcl.append('END_FUNCTION_BLOCK')
        return '\n'.join(fcl)


from fl.engine import Operator
from fl.mamdani import MamdaniRule
from fl.ruleblock import RuleBlock
from fl.variable import InputVariable, OutputVariable
import fl.term
import inspect
import re
class FCLImporter(object):
    
    def __init__(self):
        self.fe = Engine()
        #the Operator is not automatically loaded due to possibly 
        #multiple ruleblocks with different operators,
        #different defuzzifier in output variables, etc.
        self.fe.operator = None
    
    def engine(self, fcl):
        tags = {'VAR_INPUT': 'END_VAR',
                'VAR_OUTPUT': 'END_VAR',
                'FUZZIFY': 'END_FUZZIFY',
                'DEFUZZIFY': 'END_DEFUZZIFY',
                'RULEBLOCK': 'END_RULEBLOCK'}
        
        current_tag = None
        block = []
        for line in fcl.splitlines():
            #remove comments and trailing spaces
            line = re.sub(r'(^\s*)|(#.*)|(\s*$)', '', line)
            
            if len(line) == 0: continue
            
            if current_tag is not None:
                for begin_tag in tags:
                    if line.startswith(begin_tag):
                        raise SyntaxError('<%s> expected, but found <%s>' % 
                                  (current_tag[1], begin_tag))
                 
                if line.startswith(current_tag[1]):
                    self.process(current_tag[0], block)
                    current_tag = None
                    block = []
                else: 
                    block.append(line)
                
                continue
            
            if line.startswith('FUNCTION_BLOCK'):
                tokens = line.split()
                if len(tokens) > 1:
                    self.fe.name = tokens[1]
            elif line.startswith('END_FUNCTION_BLOCK'):
                break
            else:
                for begin_tag, end_tag in tags.items(): 
                    if line.startswith(begin_tag):
                        current_tag = (begin_tag, end_tag)
                        block.append(line)
                        break
                if current_tag is None:
                    raise SyntaxError('unknown block <%s>' % line)
        
        if current_tag is not None:
            if len(block) > 0:
                raise SyntaxError('%s expected for %s...' % 
                                  (current_tag[1], ' '.join(block[0:1])))
            else: raise SyntaxError('%s expected, but not found' % current_tag[1])
        return self.fe
    
    def process(self, tag, block):
        if tag == 'VAR_INPUT': self.process_input_vars(block)
        elif tag == 'VAR_OUTPUT': self.process_output_vars(block)
        elif tag == 'FUZZIFY': self.process_fuzzify(block)
        elif tag == 'DEFUZZIFY': self.process_defuzzify(block)
        elif tag == 'RULEBLOCK': self.process_ruleblock(block)
    
    def process_input_vars(self, block):
        if len(block) <= 1:
            raise SyntaxError('expected at least one variable in %s' % block)
        for variable in block[1:]:
            token = variable.split(':')
            if len(token) != 2:
                raise SyntaxError('malformed property <%s>' % variable)
            name = token[0]
            self.fe.input[name] = InputVariable(name)
    
    def process_output_vars(self, block):
        if len(block) <= 1:
            raise SyntaxError('expected at least one variable in %s' % block)
        for variable in block[1:]:
            token = variable.split(':')
            if len(token) != 2:
                raise SyntaxError('malformed property <%s>' % variable)
            name = token[0]
            self.fe.output[name] = OutputVariable(name)
    
    def extract_term(self, line):
        token = line.split(':=', 2)
        if len(token) != 2:
            raise SyntaxError('malformed property <%s>' % line)
        lvalue, rvalue = token[0], token[1].strip()
        
        token = lvalue.split()
        if len(token) != 2:
            raise SyntaxError('malformed lvalue in <%s>' % line)
        name = token[1]
        
        token = re.match(r'(\w+)\s*(\(.*\))', rvalue)
        if not token:
            raise SyntaxError('malformed rvalue in <%s>' % line)
        
        term_class = token.group(1) 
        available_terms = [term for term in dir(fl.term) 
                           if inspect.isclass(eval('fl.term.%s' % term))]
        
        if term_class not in available_terms:
            raise SyntaxError('unknown term <%s>, only %s are available'
                              % (term_class, available_terms))
        
        term_args = token.group(2)
        term_args = "'%s', %s" % (name, term_args[1:-1])  # removes parentheses
        
        term_object = None
        try:
            term_object = eval('fl.term.%s(%s)' % (term_class, term_args))
        except:
            raise
        return term_object
        
    
    def process_fuzzify(self, block):
        token = block[0].split()
        if len(token) != 2:
            raise SyntaxError('malformed block definition in <%s> ' % token)
        name = token[1]
        
        for line in block[1:]:
            term = self.extract_term(line)
            self.fe.input[name].term[term.name] = term
    
    def extract_defuzzifier(self, line):
        token = line.split(':')
        if len(token) != 2:
            raise SyntaxError('malformed property <%s>' % line)
        
        name = token[1].strip().replace(';', '')
        
        acronyms = {}
        for defuzz in dir(fl.defuzzifier):
            if not inspect.isclass(eval('fl.defuzzifier.%s' % defuzz)):
                continue
            acronym = []
            for letter in defuzz:
                if letter.isupper(): acronym.append(letter)
            acronyms[''.join(acronym)] = defuzz
        
        if name not in acronyms:
            raise SyntaxError('unknown defuzzifier <%s>, only %s are available'
                              % (name, acronyms.keys()))
        
        defuzzifier = eval('fl.defuzzifier.%s()' % acronyms[name])
        
        return defuzzifier
    
    def extract_operator(self, line):
        token = line.split(':')
        if len(token) != 2:
            raise SyntaxError('malformed property <%s>' % line)
        
        operation = token[0].strip()
        operator = token[1].strip().replace(';','')
        
        clazz = None
        if operation == 'AND': clazz = fl.operator.FuzzyAnd
        elif operation == 'OR': clazz = fl.operator.FuzzyOr
        elif operation == 'ACT': clazz = fl.operator.FuzzyActivation
        elif operation == 'ACCU': clazz = fl.operator.FuzzyAccumulation
        else:
            raise SyntaxError('unknown operation <%s> in %s' % (operation, line))
        
        import types
        methods = {}
        for method in dir(clazz):
            if method.startswith('__'): continue
            method_ref = getattr(clazz, method)
            if isinstance(method_ref, types.FunctionType):
                methods[method.upper()] = method_ref
        
        if operator not in methods:
            raise SyntaxError('unknown operator <%s> in %s' %(operator, line))
        
        return methods[operator]
            
    
    def process_defuzzify(self, block):
        token = block[0].split()
        if len(token) != 2:
            raise SyntaxError('malformed block definition in <%s> ' % token)
        name = token[1]
        
        for line in block[1:]:
            line = line.strip()
            if line.startswith('TERM'):
                term = self.extract_term(line)
                self.fe.output[name].term[term.name] = term
            elif line.startswith('METHOD'):
                defuzzifier = self.extract_defuzzifier(line)
                self.fe.output[name].defuzzifier = defuzzifier
            elif line.startswith('ACCU'):
                self.fe.output[name].output.accumulation = self.extract_operator(line)
            elif line.startswith('DEFAULT'):
                token = line.split(':')
                if len(token) != 2:
                    raise SyntaxError('malformed property <%s>' % line)
                value = None
                try:
                    value = float(token[1].strip().replace(';',''))
                except:
                    raise ValueError('invalid default value in <%s>' % line)
                self.fe.output[name].default = value
            else:
                raise SyntaxError('unknown property <%s>' % line)
    
    
    def process_ruleblock(self, block):
        ruleblock = RuleBlock()
        token = block[0].split()
        if len(token) == 2: 
            ruleblock.name = token[1].strip()
        
        for line in block[1:]:
            line = line.strip()
            if line.startswith('RULE'):
                token = line.split(':')
                if len(token) != 2:
                    raise SyntaxError('malformed property <%s>' % line)
                ruleblock.append(MamdaniRule.parse(token[1].replace(';',''), self.fe))
            elif line.startswith('AND'):
                ruleblock.tnorm = self.extract_operator(line)
            elif line.startswith('OR'):
                ruleblock.snorm = self.extract_operator(line)
            elif line.startswith('ACT'):
                ruleblock.activation = self.extract_operator(line)
            else:
                raise SyntaxError('unknown property <%s>' % line)
        
        self.fe.ruleblock[ruleblock.name] = ruleblock

if __name__ == '__main__':
    from fl.example import Example
    fe = Example().simple_mamdani()
    export = FCLExporter()
    fcl = export.engine(fe)
    print(fcl)
    
    print('\n====================\n')
    
    fe = FCLImporter().engine(fcl)
    copy_fcl = export.engine(fe) 
    print(copy_fcl)
    
    print('\n====================\n')
    
    if fcl == copy_fcl:
        print('Exporter/Importer are just FINE :)')
    else:
        print('DIFFERENT results from Exporter/Importer!!!')
    
    
    
    
    
    
    
    
    
    
    
