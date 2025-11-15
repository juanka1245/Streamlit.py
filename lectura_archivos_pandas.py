import pandas as pd  #Importamos pandas

df=pd.read_csv('titanic.csv') #Leemos el csv

#df=pd.read_excel('titanic.xlsx')  #Leemos el excel

#df=pd.read_json('titanic.json')  #Leemos el json


#print(df.head()) #Imprimimos las primeras 5 filas del data frame

#print(df.head(10)) #Imprimimos las primeras 10 filas

#print(df.tail())  #Imprimimos las últimas 5 filas


#print(df.info())  #Imprimir el tipo de dato de las columnas

#print(df['PassengerId'])   #Imprimimos la columna PassengerId

#print(df['Name'])   #Imprimimos los nombres de los pasajeros

#print(df[['Name','PassengerId']])   #Imprimimos name y PassengerId


#Imprimir unicamente los registros donde Age>=18
#mayores_18=df[df['Age']>=18]
#Imprimo los mayores a 18 años
#print(mayores_18)
#Imprimo solo la edad 
#print(mayores_18['Age'])


#Loc y el iloc

#Loc sirve para filtrar por etiquetas
#mayores_18=df.loc[df['Age']>=18]
#Imprimo los mayores a 18 años
#print(mayores_18)
#Imprimo solo la edad 
#print(mayores_18['Age'])


#iloc sirve para filtrar por el índice
#primera_fila=df.iloc[0]
#Imprimo la primera fila
#print(primera_fila)


#Doble filtro, imprimir los registros de las mujeres mayores a 18 años
#mujeres_mayores_18=df[(df['Sex']=='female') & (df['Age']>=18)]
#Imprimimos las mujeres mayores a 18
#print(mujeres_mayores_18)


#Doble filtro usando loc, imprimir los registros de las mujeres mayores a 18 años
#mujeres_mayores_18=df.loc[(df['Sex']=='female') & (df['Age']>=18)]
#Imprimimos las mujeres mayores a 18
#print(mujeres_mayores_18)


#Imprimimos las estadísticas básicas del dataframe
#print(df.describe())


#imprimir el nombre del pasajero 48
print(df.loc[47,'Name'])    

#imprimir cual es el promedio de columna age
print(df['Age'].mean().round(2)) 

#imrpimir los registro de las personas entre 20 y 35 años
Edades_entre_20y35=df.loc[(df['Age'] >= 20) & (df['Age'] <= 35)]
print(Edades_entre_20y35)

#imprimir cuantas mujeres en numero entero hay en la data
mujeres=df[df['Sex']=='female'].value_counts()
print(mujeres)


#imprimir cuantos hombres hay en la data
hombres=df[df['Sex']=='male'].value_counts()
print(hombres)

#imprirmir la edad del pasajero con PassengerId 10
print(df.loc[df['PassengerId']==10,'Age'])


#Imrpimir las estadisticas básicas del dataframe
print(df.describe())


