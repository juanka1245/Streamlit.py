import statistics

def calcular_estadisticas():
    """
    Función principal que solicita datos al usuario, calcula la Moda,
    Media y Mediana, y muestra los resultados.
    """
    
    # 1. Solicitar el nombre del usuario
    nombre_usuario = input("1. Por favor, introduce tu nombre: ")
    print(f"\n¡Hola, {nombre_usuario}!")
    
    # 2. Solicitar la cantidad de datos a ingresar
    while True:
        try:
            cantidad_datos = int(input("2. Introduce la cantidad de números enteros a ingresar: "))
            if cantidad_datos > 0:
                break
            else:
                print("La cantidad debe ser un número entero positivo. Intenta de nuevo.")
        except ValueError:
            print("Entrada inválida. Por favor, introduce un número entero.")

    # 3. Solicitar los datos y almacenarlos en un vector (lista)
    datos = []
    print(f"\n3. Introduce los {cantidad_datos} números enteros uno por uno:")
    
    for i in range(cantidad_datos):
        while True:
            try:
                # Solicitar y almacenar el dato
                dato = int(input(f"Dato #{i + 1}: "))
                datos.append(dato)
                break
            except ValueError:
                print("Entrada inválida. Por favor, introduce un número entero.")

    # 4. Calcular Moda, Media y Mediana
    # Nota: statistics.mode() fallará si no hay una moda única (ej. [1, 2, 3, 4]).
    # Usamos statistics.multimode() para manejar múltiples modas y statistics.mode() si queremos una sola.
    
    try:
        moda = statistics.mode(datos) # Calcula la moda con la librería
    except statistics.StatisticsError:
        # Si no hay moda única, indicamos que puede haber varias o ninguna.
        moda = "No hay una moda única o todos los valores son únicos."

    media = statistics.mean(datos) # Calcula el promedio (Media Aritmética)
    mediana = statistics.median(datos) # Calcula el valor central (Mediana)
    
    # 5. Mostrar un mensaje al usuario con el nombre y los resultados
    print("\n" + "="*50)
    print(f"RESUMEN ESTADÍSTICO PARA {nombre_usuario.upper()}")
    print("="*50)
    
    print("\n--- Datos Informados ---")
    print(f"Los datos ingresados son: {datos}")
    
    print("\n--- Resultados de Medidas de Tendencia Central ---")
    print(f"• Media (Promedio): {media:.2f}")
    print(f"• Mediana: {mediana}")
    print(f"• Moda: {moda}")
    
    print("\n¡Proceso completado!")
    print("="*50)

# Ejecutar el programa
if __name__ == "__main__":
    calcular_estadisticas()