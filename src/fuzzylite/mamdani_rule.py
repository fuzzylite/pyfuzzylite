'''
Created on 31/10/2012

@author: jcrada
'''

from fuzzylite.fuzzy_rule import FuzzyRule, FuzzyConsequent

class MamdaniRule(FuzzyRule):
    
    def __init__(self):
        FuzzyRule.__init__(self)
        
    def parse(self, rule):
        pass
#        'if then'
#        self.antecedent= 
        


class MamdaniConsequent(FuzzyConsequent):
    '''A Mamdani consequent of the form <variable> is [hedges] <term> [with <weight>].'''

    class Proposition:
        def __init__(self):
            self.variable = None
            self.hedges = []
            self.term = None
            self.weight = 1.0
        
        def __str__(self):
            result = [self.variable.name, FuzzyRule.FR_IS]
            result.extend([hedge.name for hedge in self.hedges])
            if self.term is not None: #if hedge == 'any', term is None
                result.append(self.term.name)
            if self.weight != 1.0:
                self.append('%s %f' % (FuzzyRule.FR_WITH, self.weight))
            return ' '.join(result)

    def __init__(self):
        self.propositions = []
        
    def __str__(self):
        return (' %s ' % FuzzyRule.FR_AND).join([str(prop) for prop in self.propositions])

    def fire(self, strength):
        import copy
        for proposition in self.propositions:
            term = copy.deepcopy(proposition.term)
            alphacut = strength * proposition.weight
            for hedge in proposition.hedges: 
                alphacut = hedge.apply(alphacut)
            term.alphacut = alphacut
            proposition.variable.output.add(term)
            

    def parse(self, infix, fuzzy_engine):
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
                if token in fuzzy_engine.input:
                    proposition.append(MamdaniConsequent.Proposition())
                    proposition[-1].variable = fuzzy_engine.input[token]
                    state = [s_is]
                    continue
            
            if s_is in state:
                if token == FuzzyRule.FR_IS:
                    state = [s_hedge, s_term]
                    continue
            
            if s_hedge in state:
                if token in fuzzy_engine.hedge:
                    proposition[-1].hedges.append(fuzzy_engine.hedge[token])
                    state = [s_hedge, s_term]
                    continue

            if s_term in state:
                if token in proposition[-1].variable.terms:
                    proposition[-1].term = proposition[-1].variable.terms[token]
                    state = [s_and, s_with]
                    continue

            if s_and in state:
                if token == FuzzyRule.FR_AND:
                    state = [s_variable]
                    continue
            
            if s_with in state:
                if token == FuzzyRule.FR_WITH:
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
                raise SyntaxError('expected variable, but found <%s>' % token)
            elif s_is in state:
                raise SyntaxError('expected keyword <%s>, but found <%s>' 
                                      % (FuzzyRule.FR_IS, token))
            elif s_hedge in state or s_term in state:
                raise SyntaxError('expected hedge or term, but found <%s>' % token)
            elif s_and in state or s_with in state:
                raise SyntaxError('expected operators <%s> or <%s>, but found <%s>' 
                                  % (FuzzyRule.FR_AND, FuzzyRule.FR_WITH, token))
            elif s_float in state:
                raise SyntaxError('expected weight magnitude, but found <%s>' % token)
            else:
                raise SyntaxError('unexpected token found ' + token)
        
        self.propositions = proposition
    

        
    
if __name__ == '__main__':
    from fuzzylite.example import Example
    fe = Example.simple_mamdani()
    mc = MamdaniConsequent()
    infix = 'Energy is LOW and Energy is very LOW and Energy is MEDIUM'
    mc.parse(infix, fe)
    print(str(mc))
    
    
    
    
    
    
    
    
    
    