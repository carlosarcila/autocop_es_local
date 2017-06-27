#!/usr/bin/python
# -*- coding: UTF-8 -*-
notas = [4, 5, 8, 10]

#Creo una función para calcultar el 60% de un número
def por60(n):
    return n*0.6
#print(por60(4))

#Creo una función para aplicar la función a una lista
def por60_L(L):
    return [x*0.6 for x in L]
#print(por60_L(notas))
#Se podría usar la función map
#print(map(por60, notas))

#Creo una función para calcultar el 60% de un número, pero con condiciones
def por60(n):
    if n <= 5:
        return n
    else:
        return n*0.6
#print(por60(5))

#Creo una función con condiciones para aplicar la función a una lista
def por60_L(L):
    return [i if i <=5 else i*0.6 for i in L]
print(por60_L(notas))
