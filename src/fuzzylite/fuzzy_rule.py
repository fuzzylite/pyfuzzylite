'''
Created on 27/10/2012

@author: jcrada
'''

class FuzzyRule:
    '''Defines a fuzzy rule'''
    
    FR_IF = 'if'
    FR_IS = 'is'
    FR_THEN = 'then'
    FR_AND = 'and'
    FR_OR = 'or'
    FR_WITH = 'with'

    def __init__(self):
        self.antecedent = None
        self.consequents = []
        


