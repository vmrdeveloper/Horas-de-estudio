import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#cargamos los datos
dataset = pd.read_csv('Horas de estudio (1).csv')

#preparamos los datos

x = dataset[['Hours']]
y = dataset[['Scores']]


#Entrenamos el modelo dividiendo la data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=53)

model = LinearRegression()
model.fit(X_train, y_train)

#Evaluando el modelo con la data que no ha visto
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Error Cuadrático Medio (MSE): {mse}")
print(f"Coeficiente de Determinación (R^2): {r2}")

#visualizar resultados
plt.scatter(X_train, y_train, color='blue', label='Datos de Entrenamiento')
plt.plot(X_train, model.predict(X_train), color='red', label='Línea de Regresión')
plt.xlabel('Horas de Estudio')
plt.ylabel('Calificación')
plt.title('Regresión Lineal - Calificación vs Horas de Estudio')
plt.legend()
plt.show()

#Visualizar resultados con la data que no entreno
plt.scatter(X_test, y_test, color='green', label='Datos de prueba')
plt.plot(X_train, model.predict(X_train), color='red', label='Línea de Regresión')
plt.xlabel('Horas de Estudio')
plt.ylabel('Calificación')
plt.title('Regresión Lineal - Calificación vs Horas de Estudio')
plt.legend()
plt.show()

nuevas_horas = np.array([[7.8]])  
prediccion = model.predict(nuevas_horas)
print(f'Segun las horas que colocaste se predice que tus notas seran: {prediccion[0]}')
