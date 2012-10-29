'''
Created on 27/10/2012

@author: jcrada
'''

from fuzzylite.fuzzy_operator import FuzzyOperator
from fuzzylite import hedge_set
from fuzzylite.fuzzy_rule import FuzzyRule
 

class FuzzyAntecedent:
    '''Defines the antecedent of a fuzzy rule'''

    def __init__(self, fuzzy_operator = FuzzyOperator.Default()):
        self.input_var = None
        self.fuzzy_operator = fuzzy_operator
    
    def degree_of_truth(self):
        raise NotImplementedError()


class DescriptiveAntecedent(FuzzyAntecedent):
    '''Defines a descriptive antecedent'''
    
    class Node: Terminal, And, Or = range(3)
    
    def __init__(self, fuzzy_operator=FuzzyOperator.Default()):
        FuzzyAntecedent.__init__(self, fuzzy_operator)
        self.left = None
        self.right = None
        self.node = None
        self.hedges = []
        self.term = None
        
    def __str__(self):
        result = ''
        if self.node is self.Node.Terminal:
            result += self.input_var.name + ' ' + FuzzyRule.FR_IS + ' ' + \
                        ' '.join(hedge[0] for hedge in self.hedges)
            result += self.term.name
        else:    
            result += ' ( ' + str(self.left)
            if self.node == self.Node.And: result += FuzzyRule.FR_AND
            elif self.node == self.Node.Or: result += FuzzyRule.FR_OR
            else: raise ValueError('node cannot be None')
            result += ' ' + str(self.right) + ' ) '
        return result
    
    def degree_of_truth(self):
        #if node is terminal, then return the membership value and its hedges
        if self.node is self.Node.Terminal:
            result = self.term.membership(self.input_var.input)
            for hedge in self.hedges: result = hedge(result)
            return result
        else:
            if not self.left  or not self.right:
                raise ValueError('left and right antecedents must exist')
            
            if self.node is self.Node.And:
                return self.fuzzy_operator.tnorm(self.left.degree_of_truth(), 
                                                 self.right.degree_of_truth())
            elif self.node is self.Node.Or:
                return self.fuzzy_operator.snorm(self.left.degree_of_truth(), 
                                                 self.right.degree_of_truth())
            else: raise ValueError('node cannot be None')

    
        
        
        
        
        
        
        


if __name__ == '__main__':
    pass

















