import pandas as pd

# Datos de ejemplo: una lista de gastos
# Puedes reemplazar estos valores con tus propios datos reales
gastos_lista = [1500, 3200, 850, 4100, 1950, 2150, 900, 5000]

# --- 1- Imprimir una serie con los gastos ---
# Crear la Serie de Pandas a partir de la lista
gastos_serie = pd.Series(gastos_lista, name="Gastos Mensuales")
print("1. Serie con todos los gastos:")
print(gastos_serie)
print("-" * 30)

# --- 2- Imprimir los gastos más altos ---
# Usamos el método .max() para encontrar el gasto más alto
gasto_maximo = gastos_serie.max()
print("2. El gasto más alto es:")
print(f"${gasto_maximo}")
# Si quisieras ver los 2 o 3 gastos más altos, podrías usar:
# gastos_serie.nlargest(3)
print("-" * 30)

# --- 3- Imprimir la suma de todos los gastos ---
# Usamos el método .sum() para obtener la suma total
suma_gastos = gastos_serie.sum()
print("3. La suma de todos los gastos es:")
print(f"${suma_gastos}")
print("-" * 30)

# --- 4- Imprimir los gastos mayores a 2000 ---
# Usamos indexación booleana (filtrado) para seleccionar los valores > 2000
gastos_mayores_2000 = gastos_serie[gastos_serie > 2000]
print("4. Gastos mayores a 2000:")
print(gastos_mayores_2000)