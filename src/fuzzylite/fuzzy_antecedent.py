'''
Created on 27/10/2012

@author: jcrada
'''

from fuzzylite.fuzzy_operator import FuzzyOperator
from fuzzylite import hedge_dict
from fuzzylite.fuzzy_rule import FuzzyRule
 

class FuzzyAntecedent:
    '''Defines the antecedent of a fuzzy rule'''

    def __init__(self, fuzzy_operator = FuzzyOperator.Default()):
        self.input = None
        self.fuzzy_operator = fuzzy_operator
    
    def degree_of_truth(self):
        raise NotImplementedError()


class DescriptiveAntecedent(FuzzyAntecedent):
    '''Defines a descriptive antecedent'''
    
    def __init__(self, fuzzy_operator=FuzzyOperator.Default()):
        FuzzyAntecedent.__init__(self, fuzzy_operator)
        self.left = None
        self.right = None
        self.operator = None
        self.hedge = []
        self.term = None
        
    def __str__(self):
        result = ''
        if self.operator is None:
            result += self.input.name + ' ' + FuzzyRule.FR_IS + ' ' + \
                        ' '.join(hedge[0] for hedge in self.hedge)
            result += self.term.name
        else:    
            result += ' ( ' + str(self.left)
            if self.operator == 'and': result += FuzzyRule.FR_AND
            elif self.operator == 'or': result += FuzzyRule.FR_OR
            else: raise ValueError('unknown operator: ' + self.operator)
            result += ' ' + str(self.right) + ' ) '
        return result
    
    def degree_of_truth(self):
        #if operator is terminal, then return the membership value and its hedge
        if self.operator is None:
            result = self.term.membership(self.input.input)
            for hedge in self.hedge: 
                result = hedge(result)
            return result
        else:
            if not self.left  or not self.right:
                raise ValueError('left and right antecedents must exist')
            
            if self.operator == 'and':
                return self.fuzzy_operator.tnorm(self.left.degree_of_truth(), 
                                                 self.right.degree_of_truth())
            elif self.operator == 'or':
                return self.fuzzy_operator.snorm(self.left.degree_of_truth(), 
                                                 self.right.degree_of_truth())
            else: raise ValueError('unknown operator: ' + self.operator)



if __name__ == '__main__':
    pass

















