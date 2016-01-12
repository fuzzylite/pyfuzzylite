'''
Created on 31/10/2012

@author: jcrada
'''

from fl.rule import Rule, FuzzyAntecedent, FuzzyConsequent 
from fl.parser import Parser
import re
class MamdaniRule(Rule):
    
    def __init__(self):
        Rule.__init__(self)
    
    @classmethod
    def parse(cls, rule, fe):
        '''
        Parses a fuzzy rule from text.
        
        rule -- fuzzy rule in format <if ... then ...>
        fe -- an instance to the fuzzy engine
        '''
        
        matcher = re.compile('(^\s*if\s+)(.*)(\s+then\s+)(.*)').match(rule)
        if not matcher or len(matcher.groups()) != 4:
            raise SyntaxError('expected rule as <%s ... %s ...>, but found <%s>'
                              % (Rule.FR_IF,Rule.FR_THEN, rule))
        
        #matcher.groups() = ('if ', 'Energy is LOW', ' then ', 'Health is BAD')
        #matcher.group(0) is whole rule
        antecedent = matcher.group(2)
        consequent = matcher.group(4)
        
        instance = cls()
        
        instance.antecedent = MamdaniAntecedent()
        instance.antecedent.parse(antecedent, fe)
        
        instance.consequent = MamdaniConsequent() 
        instance.consequent.parse(consequent, fe)
        
        return instance



class MamdaniAntecedent(FuzzyAntecedent):
    '''A Fuzzy Antecedent as an expression tree.'''
    def __init__(self):
        self.root = None
    
    class Proposition:
        def __init__(self):
            self.variable = None
            self.hedges = []
            self.term = None
        
        def __str__(self):
            result = [self.variable.name, Rule.FR_IS]
            result.extend([hedge.name for hedge in self.hedges])
            if self.term is not None: #if hedge == 'any', term is None
                result.append(self.term.name)
            return ' '.join(result) 
            
    class Operator:
        def __init__(self, operator=None):
            self.operator = operator
            self.left = None
            self.right = None

        def __str__(self):
            return str(self.operator)
    
    
    def firing_strength(self, tnorm, snorm, node = None):
        if node is None: 
            node = self.root
        if isinstance(node, MamdaniAntecedent.Proposition):
            result = node.term.membership(node.variable.input)
            for hedge in node.hedges:
                result = hedge.apply(result)
            return result
        elif isinstance(node, MamdaniAntecedent.Operator):
            if not (node.left or node.right):
                raise ValueError('left and right operands must exist')
            if node.operator == Rule.FR_AND:
                return tnorm(self.firing_strength(tnorm, snorm, node=node.left),
                             self.firing_strength(tnorm, snorm, node=node.right))
                #return tnorm(self.firing_strength(tnorm, snorm, node=self.left),  # TODO:  Is the previous change right????
                #            self.firing_strength(tnorm, snorm, node=self.right))
            elif node.operator == Rule.FR_OR:
                return snorm(self.firing_strength(tnorm, snorm, node=self.left),
                             self.firing_strength(tnorm, snorm, node=self.right))
            else: raise ValueError('unknown operator %s' % node.operator)
        else: raise TypeError('unexpected node type %s' % type(node))
        
        
    
    def parse(self, infix, engine, operators=Parser.default_operators):
        '''
        Builds an proposition tree from the antecedent of a fuzzy rule.
        The rules are:  1) After a variable comes 'is',
                        2) After 'is' comes a hedge or a term
                        3) After a hedge comes a hedge or a term
                        4) After a term comes a variable or an operator
        '''
        postfix = Parser.infix_to_postfix(infix, operators, {})
        #//e.g. Postfix antecedent: Energy is LOW Distance is FAR_AWAY and
        #3 4 2 * 1 5 - 2 3 ^ ^ / +
        tokens = postfix.split()
        s_variable, s_is, s_hedge, s_term, s_operator = range(5)
        state = [s_variable]
        proposition = []
        for token in tokens:
            if s_variable in state:
                if token in engine.input:
                    proposition.append(MamdaniAntecedent.Proposition())
                    proposition[-1].variable = engine.input[token]
                    state = [s_is]
                    continue
            
            if s_is in state:
                if token != Rule.FR_IS:
                    raise SyntaxError('expected keyword <%s>, but found <%s>' 
                                      % (Rule.FR_IS, token))
                state = [s_hedge, s_term]
                continue
            
            if s_hedge in state:
                if token in engine.hedge:
                    proposition[-1].hedges.append(engine.hedge[token])
                    if token == 'any':
                        state = [s_variable, s_operator]
                    else:
                        state = [s_hedge, s_term]
                    continue

            if s_term in state:
                if token in proposition[-1].variable.term:
                    proposition[-1].term = proposition[-1].variable.term[token]
                    state = [s_variable, s_operator]
                    continue

            if s_operator in state:
                if token in operators:
                    operator_node = MamdaniAntecedent.Operator(token)
                    if len(proposition) < 2:
                        raise SyntaxError('operator <%s> expected 2 operands, '
                                          'but found just %i' % (token, len(proposition)))
                    operator_node.right = proposition.pop()
                    operator_node.left = proposition.pop()
                    proposition.append(operator_node)
                    state = [s_variable, s_operator]
                    continue
            #If reached this point, there was an error
            if s_variable in state or s_operator in state:
                raise SyntaxError('expected input variable or operator, but found <%s>' % token)
            elif s_hedge in state or s_term in state:
                raise SyntaxError('expected hedge or term, but found <%s>' % token)
            else:
                raise SyntaxError('unexpected token <%s>' % token)
        
        if len(proposition) != 1:
            raise ValueError('stack expected to contain the root, but contains %i ' 
                             'expressions' % len(proposition))
        self.root = proposition.pop()

    def str_prefix(self, node=None):
        if node is None: node = self.root
        if isinstance(node, MamdaniAntecedent.Proposition):
            return str(node)
        result = []
        result.append(str(node))
        result.append(self.str_prefix(node=node.left))
        result.append(self.str_prefix(node=node.right))
        return ' '.join(result)
    
    def str_infix(self, node=None):
        if node is None: node = self.root
        if isinstance(node, MamdaniAntecedent.Proposition):
            return str(node)
        result = []
        result.append(self.str_infix(node=node.left))
        result.append(str(node))
        result.append(self.str_infix(node=node.right))
        return ' '.join(result)
     
    def str_postfix(self, node=None):
        if node is None: node = self.root
        if isinstance(node, MamdaniAntecedent.Proposition):
            return str(node)
        result = []
        result.append(self.str_postfix(node=node.left))
        result.append(self.str_postfix(node=node.right))
        result.append(str(node))
        return ' '.join(result)

    def __str__(self):
        return self.str_postfix()

from fl.term import Output
class MamdaniConsequent(FuzzyConsequent):
    '''A Mamdani consequent of the form <variable> is [hedges] <term> [with <weight>].'''

    class Proposition:
        def __init__(self):
            self.variable = None
            self.hedges = []
            self.term = None
            self.weight = 1.0
        
        def __str__(self):
            result = [self.variable.name, Rule.FR_IS]
            result.extend([hedge.name for hedge in self.hedges])
            if self.term is not None: #if hedge == 'any', term is None
                result.append(self.term.name)
            if self.weight != 1.0:
                self.append('%s %f' % (Rule.FR_WITH, self.weight))
            return ' '.join(result)

    def __init__(self):
        FuzzyConsequent.__init__(self)
        self.propositions = []
        
    def __str__(self):
        return (' %s ' % Rule.FR_AND).join([str(prop) for prop in self.propositions])

    def fire(self, strength, activation):
        self.logger.debug('Firing at %s Rule: %s' % (strength, self))
            
        for proposition in self.propositions:
            term = Output(proposition.term)
            alphacut = strength * proposition.weight
            for hedge in proposition.hedges: 
                alphacut = hedge.apply(alphacut)
            term.alphacut = alphacut
            term.activation = activation
            proposition.variable.output.append(term)
            

    def parse(self, infix, engine):
        '''
        Extracts the list of propositions from the consequent
        The rules are:  1) After a variable comes 'is',
                        2) After 'is' comes a hedge or a term
                        3) After a hedge comes a hedge or a term
                        4) After a term comes operators 'and' or 'with'
                        5) After operator 'and' comes a variable
                        6) After operator 'with' comes a float 
        '''
        tokens = infix.split()
        s_variable, s_is, s_hedge, s_term, s_and, s_with, s_float= range(7)
        state = [s_variable]
        proposition = []
        for token in tokens:
            if s_variable in state:
                if token in engine.output:
                    proposition.append(MamdaniConsequent.Proposition())
                    proposition[-1].variable = engine.output[token]
                    state = [s_is]
                    continue
            
            if s_is in state:
                if token == Rule.FR_IS:
                    state = [s_hedge, s_term]
                    continue
            
            if s_hedge in state:
                if token in engine.hedge:
                    proposition[-1].hedges.append(engine.hedge[token])
                    state = [s_hedge, s_term]
                    continue

            if s_term in state:
                if token in proposition[-1].variable.term:
                    proposition[-1].term = proposition[-1].variable.term[token]
                    state = [s_and, s_with]
                    continue

            if s_and in state:
                if token == Rule.FR_AND:
                    state = [s_variable]
                    continue
            
            if s_with in state:
                if token == Rule.FR_WITH:
                    state = [s_float]
                    continue
                
            if s_float in state:
                try:
                    proposition[-1].weight = float(token)
                    state = [s_and]
                    continue
                except ValueError:
                    pass
                
            
            #If reached this point, there was an error
            if s_variable in state:
                raise SyntaxError('expected output variable, but found <%s>' % token)
            elif s_is in state:
                raise SyntaxError('expected keyword <%s>, but found <%s>' 
                                      % (Rule.FR_IS, token))
            elif s_hedge in state or s_term in state:
                raise SyntaxError('expected hedge or term, but found <%s>' % token)
            elif s_and in state or s_with in state:
                raise SyntaxError('expected operators <%s> or <%s>, but found <%s>' 
                                  % (Rule.FR_AND, Rule.FR_WITH, token))
            elif s_float in state:
                raise SyntaxError('expected weight magnitude, but found <%s>' % token)
            else:
                raise SyntaxError('unexpected token found ' + token)
        
        self.propositions = proposition

if __name__ == '__main__':
    from fl.example import Example
    fe = Example().simple_mamdani()
    infix = 'if Energy is LOW then Health is BAD'
    rule = MamdaniRule.parse(infix, fe)
    print(rule) 
    
    
    
    
    
    
    
    
    
    
