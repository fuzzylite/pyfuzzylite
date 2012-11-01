'''
Created on 28/10/2012

@author: jcrada
'''

from math import *

class Operator:
    def __init__(self, token, precedence, mask=None, arity=2, associativity= -1):
        self.token = token
        self.precedence = precedence
        self.arity = arity
        self.associativity = associativity
        self.mask = mask if mask else token #to prevent alpha operators split within operands
    
    @staticmethod
    def default_operators():
        p = 7
        from fuzzylite.fuzzy_rule import FuzzyRule
        o_and = FuzzyRule.FR_AND
        o_or = FuzzyRule.FR_OR
        return ({
               '!':Operator('!', p, arity=1), '~':Operator('~', p, arity=1),
               '^':Operator('^', p - 1, associativity=1),
               '*':Operator('*', p - 2), '/':Operator('/', p - 2), '%':Operator('%', p - 2),
               '+':Operator('+', p - 3), '-':Operator('-', p - 3),
               '&':Operator('&', p - 4), '|':Operator('|', p - 5),
               '&&':Operator('&&', p - 6), o_and:Operator(o_and, p - 6, mask=' %s ' % o_and),
               '||':Operator('||', p - 7), o_or:Operator(o_or, p - 7, mask=' %s ' % o_or)
                })

class Function:
    def __init__(self, token, arity, associativity= -1):
        self.token = token
        self.arity = arity
        self.associativity = associativity

    @staticmethod
    def default_functions():
        import inspect, math
        functions = {f:Function(f, 1) for f in dir(math) if inspect.isbuiltin(eval(f))}
        # updating those functions with 2 parameters manually as builtin cannot be inspected
        del functions['fsum']  # deletes fsum as parameter is iterable 
        twoargs = ['copysign', 'fmod', 'ldexp', 'log', 'pow', 'atan2', 'hypot']
        for f in twoargs: functions[f].arity = 2
        return functions


class Parser:
    
    
    default_operators = Operator.default_operators()
    default_functions = Function.default_functions()

    @staticmethod
    def infix_to_postfix(infix, operators=default_operators,
                         functions=default_functions):
        '''
        Converts from infix notation to postfix using the Shunting yard algorithm
        as described in http://en.wikipedia.org/w/index.php?title=Shunting-yard_algorithm&oldid=516997362
        '''
        separators = sorted([o.mask for o in operators.values()] + ['(', ')', ','],
                          reverse=True)
        #separate is sorted such that ops like && be first to be separated instead of &
        import re
        regex = '|'.join([re.escape(sep) for sep in separators])
        infix = re.sub('(' + regex + ')' , r' \1 ', infix)
        tokens = infix.split()
        from collections import deque
        queue = deque()
        stack = []
        is_operand = lambda x: not (x in operators or x in functions or x in ('(', ')'))
        for token in tokens:
            if is_operand(token):
                queue.append(token)
            elif token in functions:
                stack.append(token)
            elif token == ',':
                while stack and stack[-1] != '(':
                    queue.append(stack.pop())
                if not stack or stack[-1] != '(':
                    raise SyntaxError('mismatching parentheses in: ' + infix)
            elif token in operators:
                o1 = operators[token]
                while True:
                    o2 = None
                    if stack and stack[-1] in operators:
                        o2 = operators[stack[-1]]
                    else: break
                    if (o1.associativity < 0 and o1.precedence <= o2.precedence
                        or o1.precedence < o2.precedence):
                        queue.append(stack[-1])
                        stack.pop()
                    else: break
                stack.append(token)
            elif token == '(':
                stack.append(token)
            elif token == ')':
                while stack and stack[-1] != '(':
                    queue.append(stack.pop())
                if not stack or stack[-1] != '(':
                    raise SyntaxError('mismatching parentheses in: ' + infix)
                stack.pop()
                if stack and stack[-1] in functions:
                    queue.append(stack.pop())
            else: raise AssertionError('this should have never occurred')
        
        while stack:
            if stack[-1] in ('(', ')'):
                raise SyntaxError('mismatching parentheses in: ' + infix)
            queue.append(stack.pop())

        return ' '.join(queue)

 
        

if __name__ == '__main__':
    infix = '3 + 4 * 2 / ( 1 - 5 ) ^ 2 ^ 3'
    print(Parser.infix_to_postfix(infix))
    
    infix = '3+4*2/(1-5)^2^3'
    print(Parser.infix_to_postfix(infix))
    
    infix = '''(Temperature is High and Oxigen is Low) or
            (Temperature is Low and (Oxigen is Low or Oxigen is High))'''
    print(Parser.infix_to_postfix(infix))
    
    infix = 'sin(y*x)^2/x'
    print(Parser.infix_to_postfix(infix))
    
#    infix = 'sin(y*x))^2/x'
#    x = Parser.infix_to_postfix(infix)
#    print(x)
    infix = 'sin(y*x)^2/x'
    print(Parser.infix_to_postfix(infix))
    
    infix = 'sin(y,x,z,a)^2/x'
    print(Parser.infix_to_postfix(infix))




















