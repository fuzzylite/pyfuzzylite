'''
Created on 10/10/2012

@author: jcrada
'''


def Scale(x, to_min, to_max, from_min, from_max):
    '''Scales number x in range [from_min, from_max] to its equivalent in range [to_min, to_max].'''
    return (to_max - to_min) / (from_max - from_min) * (x - from_min) + to_min