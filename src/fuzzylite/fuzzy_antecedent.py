'''
Created on 30/10/2012

@author: jcrada
'''

from fuzzylite.parser import Parser
from fuzzylite.fuzzy_rule import FuzzyRule


class FuzzyAntecedent:
    '''A Fuzzy Antecedent as an expression tree.'''
    def __init__(self):
        self.root = None
    
    class Proposition:
        def __init__(self):
            self.variable = None
            self.hedges = []
            self.term = None
        
        def __str__(self):
            result = [self.variable.name, FuzzyRule.FR_IS]
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
    
    def degree_of_truth(self, fuzzy_operator, node=None):
        if node is None: 
            node = self.root
        if isinstance(node, FuzzyAntecedent.Proposition):
            
            result = node.term.membership(node.variable.input)
            for hedge in node.hedges:
                result = hedge.apply(result)
            return result
        elif isinstance(node, FuzzyAntecedent.Operator):
            if not (node.left or node.right):
                raise ValueError('left and right operands must exist')
            if node.operator == FuzzyRule.FR_AND:
                return fuzzy_operator.tnorm(self.degree_of_truth(node=self.left),
                                              self.degree_of_truth(node=self.right))
            elif node.operator == FuzzyRule.FR_OR:
                return fuzzy_operator.snorm(self.degree_of_truth(node=self.left),
                                              self.degree_of_truth(node=self.right))
            else: raise ValueError('unknown operator %s' % node.operator)
        else: raise TypeError('unexpected node type %s' % type(node))
        
        
    
    def parse(self, infix, fuzzy_engine, operators=Parser.default_operators):
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
                if token in fuzzy_engine.input:
                    proposition.append(FuzzyAntecedent.Proposition())
                    proposition[-1].variable = fuzzy_engine.input[token]
                    state = [s_is]
                    continue
            
            if s_is in state:
                if token != FuzzyRule.FR_IS:
                    raise SyntaxError('expected keyword <%s>, but found <%s>' 
                                      % (FuzzyRule.FR_IS, token))
                state = [s_hedge, s_term]
                continue
            
            if s_hedge in state:
                if token in fuzzy_engine.hedge:
                    proposition[-1].hedges.append(fuzzy_engine.hedge[token])
                    if token == 'any':
                        state = [s_variable, s_operator]
                    else:
                        state = [s_hedge, s_term]
                    continue

            if s_term in state:
                if token in proposition[-1].variable.terms:
                    proposition[-1].term = proposition[-1].variable.terms[token]
                    state = [s_variable, s_operator]
                    continue

            if s_operator in state:
                if token in operators:
                    operator_node = FuzzyAntecedent.Operator(token)
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
                raise SyntaxError('expected variable or operator, but found <%s>' % token)
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
        if isinstance(node, FuzzyAntecedent.Proposition):
            return str(node)
        result = []
        result.append(str(node))
        result.append(self.str_prefix(node=node.left))
        result.append(self.str_prefix(node=node.right))
        return ' '.join(result)
    
    def str_infix(self, node=None):
        if node is None: node = self.root
        if isinstance(node, FuzzyAntecedent.Proposition):
            return str(node)
        result = []
        result.append(self.str_infix(node=node.left))
        result.append(str(node))
        result.append(self.str_infix(node=node.right))
        return ' '.join(result)
     
    def str_postfix(self, node=None):
        if node is None: node = self.root
        if isinstance(node, FuzzyAntecedent.Proposition):
            return str(node)
        result = []
        result.append(self.str_postfix(node=node.left))
        result.append(self.str_postfix(node=node.right))
        result.append(str(node))
        return ' '.join(result)

    def __str__(self):
        return self.str_postfix()

if __name__ == '__main__':
    from fuzzylite.example import Example
    fe = Example.simple_mamdani()
    ep = FuzzyAntecedent()
    infix = ('Energy is LOW and (Energy is MEDIUM or Energy is HIGH and ' 
                                 'Energy is LOW) or Energy is MEDIUM')
#    infix = 'Energy is very any LOW and Energy is somewhat very MEDIUM'
    ep.parse(infix, fe)
    print(ep)
    print('prefix: %s' % ep.str_prefix())
    print('infix: %s' % ep.str_infix())
    print('postfix: %s' % ep.str_postfix())
#    print(fe.toFCL())
    
    















