'''
Created on 28/10/2012

@author: jcrada
'''


class Operator:

    def __init__(self, token, precedence, arity=2, associativity= -1):
        self.token = token
        self.precedence = precedence
        self.arity = arity
        self.associativity = associativity

    @staticmethod
    def default_operators():
        p = 4
        return   ({
                 '^':Operator('^', p, associativity=1),
                 '*':Operator('*', p-1), '/':Operator('/', p-1), '%':Operator('%', p-1),
                 '+':Operator('+', p-2), '-':Operator('-', p-2),
                  '&&':Operator('&&', p-3), '||':Operator('||', p-4)
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
        del functions['fsum']  # deleting fsum as parameter is iterable 
        twoargs = ['copysign', 'fmod', 'ldexp', 'log', 'pow', 'atan2', 'hypot']
        for f in twoargs: functions[f].arity = 2
        return functions

class ShuntingYardAlgorithm:
    '''Converts from infix notation to postfix.'''

    default_operators = Operator.default_operators()
    default_functions = Function.default_functions()


    def __init__(self):
        pass

    @staticmethod
    def infix_to_postfix(infix, operators=default_operators,
                         functions=default_functions):
        if operators:
            
            infix = re.sub('|'.join(operators.keys()))
        import re
        infix = re.sub(r'(\(|\)|,)' , r' \1 ', infix)
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
            else: raise AssertionError('This should have never occurred')

        while stack:
            if stack[-1] in ('(', ')'):
                raise SyntaxError('mismatching parentheses in: ' + infix)
            queue.append(stack.pop())

        return queue


if __name__ == '__main__':
    infix = '3 + 4 * 2 / ( 1 - 5 ) ^ 2 ^ 3'
    x = ShuntingYardAlgorithm.infix_to_postfix(infix)
    print(' '.join(x))
    infix = '''(Temperature is High and Oxigen is Low) or 
            (Temperature is Low and (Oxigen is Low or Oxigen is High))'''
    x = ShuntingYardAlgorithm.infix_to_postfix(infix)
    print(' '.join(x))
    infix = 'sin(y*x)^2/x'
    x = ShuntingYardAlgorithm.infix_to_postfix(infix)
    print(' '.join(x))




















