#importamos pandas
import pandas as pd
#series apartir de una lista

#series
series=pd.Series([1,2,3,4,5])
#imprimimos la serie
print(series)


#series con indices personalizados
series2=pd.Series([1,2,3,4,5], index=['a','b','c','d','e'])
#IMPRIMIMOS LA SERIEs
print(series2)  

#series a partir de un diccionario
seriediccionario=pd.Series({'a':2,'b':3,'c':4,'d':5})
print(seriediccionario)

#imprimir solo los valores de la serie
filtro=seriediccionario[seriediccionario>=3]
print(filtro)

#acceder al valor de 3 de la serie
print(seriediccionario['b'])

#imprimir la media de la serie  
print(seriediccionario.mean())

#imprimir el valor maximo de la serie
print(seriediccionario.max())

#imprimir el valor minimo de la serie
print(seriediccionario.min())

#usar la funcion personalizada en la serie
multiplo3=seriediccionario.apply(lambda x: x*3)
print(multiplo3)

