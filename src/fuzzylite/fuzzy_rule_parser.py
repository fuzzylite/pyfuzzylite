'''
Created on 27/10/2012

@author: jcrada
'''
from fuzzylite.fuzzy_rule import FuzzyRule

class FuzzyRuleParser:
    '''Contains parser methods for fuzzy rules.'''
    precedence = {'*':1, '/':1, '%':1, 
                  '+':2, '-':2,
                  'and':3, 'or':4}
    @staticmethod
    def infix_to_postfix(infix, operators = {}, precedence = {}, functions = {}): 
        '''Shunting-yard algorithm'''
        import re
        infix = re.sub(r'(\(|\))',r' \1 ', infix)
        tokens = infix.split()
        postfix = ''
        queue = []
        stack = []
        
        for token in tokens:
            if ',' in token:
                pass #deal with parameters of a function
            if token in functions:
                stack.append(token)
            else:
                queue.append(token)
        return postfix
        
if __name__ == '__main__':
    infix = '1 and 2 and 3 or 4 or 5 and 6 or (7 and 8 and 9) or 10'
    print(FuzzyRuleParser.infix_to_postfix(infix))
    
    
    