'''
Created on 30/10/2012

@author: jcrada
'''

from fuzzylite.parser import Parser
from fuzzylite.fuzzy_rule import FuzzyRule


class ExpressionTree:
    '''An expression tree.'''
    def __init__(self):
        self.root = None
    
    class Operand:
        def __init__(self):
            self.variable = None
            self.hedges = []
            self.term = None
        
        def __str__(self):
            return ' '.join([self.variable.name, FuzzyRule.FR_IS,
                             ' '.join([hedge.name for hedge in self.hedges]),
                             self.term.name]) 
            
    class Operator:
        def __init__(self, operator=None):
            self.operator = operator
            self.left = None
            self.right = None

        def __str__(self):
            return str(self.operator)
    
    def from_antecedent(self, infix, fuzzy_engine, operators=Parser.default_operators):
        '''
        Builds an expression tree from the antecedent of a fuzzy rule.
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
        expression = []
        for token in tokens:
            if s_variable in state:
                if token in fuzzy_engine.input:
                    expression.append(ExpressionTree.Operand())
                    expression[-1].variable = fuzzy_engine.input[token]
                    state = [s_is]
                    continue
            
            if s_is in state:
                if token != FuzzyRule.FR_IS:
                    raise SyntaxError('expected keyword ' + FuzzyRule.FR_IS + 
                                      ', but found ' + token)
                state = [s_hedge, s_term]
                continue
            
            if s_hedge in state:
                if token in fuzzy_engine.hedge:
                    expression[-1].hedges.append(fuzzy_engine.hedge[token])
                    state = [s_hedge, s_term]
                    continue

            if s_term in state:
                if token in expression[-1].variable.terms:
                    expression[-1].term = expression[-1].variable.terms[token]
                    state = [s_variable, s_operator]
                    continue

            if s_operator in state:
                if token in operators:
                    operator_node = ExpressionTree.Operator(token)
                    if len(expression) < 2:
                        raise SyntaxError('operator %s expected 2 operands, '
                                          'but found just %i' % (token, len(expression)))
                    operator_node.right = expression.pop()
                    operator_node.left  = expression.pop()
                    expression.append(operator_node)
                    state = [s_variable, s_operator]
                    continue
            #If reached this point, there was an error
            if s_variable in state or s_operator in state:
                raise SyntaxError('expected variable or operator, but found ' + token)
            elif s_hedge in state or s_term in state:
                raise SyntaxError('expected hedge or term, but found ' + token)
            else:
                raise SyntaxError('unexpected token found ' + token)
        
        if len(expression) != 1:
            raise ValueError('stack expected to contain the root, but contains %i ' 
                             'expressions' % len(expression))
        self.root = expression.pop()

    def str_prefix(self, node=None):
        if node is None: node = self.root
        if isinstance(node, ExpressionTree.Operand):
            return str(node)
        result = []
        result.append(str(node))
        result.append(self.str_prefix(node=node.left))
        result.append(self.str_prefix(node=node.right))
        return ' '.join(result)
    
    def str_infix(self, node=None):
        if node is None: node = self.root
        if isinstance(node, ExpressionTree.Operand):
            return str(node)
        result = []
        result.append(self.str_infix(node=node.left))
        result.append(str(node))
        result.append(self.str_infix(node=node.right))
        return ' '.join(result)
     
    def str_postfix(self, node=None):
        if node is None: node = self.root
        if isinstance(node, ExpressionTree.Operand):
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
    ep = ExpressionTree()
    infix = ('Energy is LOW and (Energy is MEDIUM or Energy is HIGH and ' 
                                 'Energy is LOW) or Energy is MEDIUM')
#    infix = 'Energy is very any LOW and Energy is somewhat very MEDIUM'
    ep.from_antecedent(infix, fe)
    print(ep)
    print('prefix: %s' % ep.str_prefix())
    print('infix: %s' % ep.str_infix())
    print('postfix: %s' % ep.str_postfix())
#    print(fe.toFCL())
    
    















