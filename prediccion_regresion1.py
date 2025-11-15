from sklearn.model_selection import train_test_split #Dividir los datos de entrenamiento y prueba
from sklearn.linear_model import LinearRegression #Modelo de regresión linealpip install numpy
import numpy as np #Librería para trbajar con operaciones matemáticas matriciales


#Crear los datos
#Variable independiente, años de experiencia del trabajador
X=np.array([1,2,3,4,5,6,7,8,9,10]).reshape(-1,1) 
#Variable dependiente(predictora), salario
y=np.array([30,32,35,37,40,42,45,47,50,52]) 


#División en entrenamiento(80%) y prueba(20%)
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2)

#Entrenamiento del modelo
modelo=LinearRegression()
modelo.fit(X_train,y_train)

#Testiar el modelo
y_pred=modelo.predict(X_test)

#Evaluar el modelo
from sklearn.metrics import mean_squared_error

print(f"MSE(Error cuadrático medio): {mean_squared_error(y_test,y_pred):.2f}")

#Nueva predicción
prediccion=modelo.predict([[11]])
print(f"Para 11 años de experiencia, el salario predicho es: {prediccion}")

#Gráfica de evaluación
import matplotlib.pyplot as plt
plt.scatter(X_train,y_train,color='blue',label='Entrenamiento')
plt.scatter(X_test,y_test,color='green',label='Prueba')
plt.plot(X,modelo.predict(X),color='red',label='Regresión')
plt.legend()
plt.xlabel("Años de experiencia")
plt.ylabel("Salario")
plt.title("Regresión lineal")
plt.savefig('regression_plot.png')
plt.show()
