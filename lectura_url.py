import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
import string


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

def consumirURL(urlAVisitar):
    try:

        url =  urlAVisitar #"https://talentotech2.com.co/"

        response = requests.get(url)
        response.raise_for_status()  # Lanza un error para códigos de estado HTTP incorrectos
        soup = BeautifulSoup(response.text, 'html.parser')

        #print(soup)


        #  Encontrar todos los elementos <p> (parrafos de html)
        # El método find_all() es el más común para buscar todas las ocurrencias de una etiqueta.
        parrafos = soup.find_all('p')

        #print(parrafos)

        # Extraer el texto de cada elemento <p> y devolverlo en una lista
        parrafos_texto = [parrafo.get_text() for parrafo in parrafos]

        #print(parrafos_texto)
        separador = " "
        todasLasPalabrasJuntas = separador.join(parrafos_texto)

        print(todasLasPalabrasJuntas)
        

        todasLasPalabrasJuntasTokenizadas = nltk.word_tokenize(todasLasPalabrasJuntas.lower(), language = 'spanish')
        resultadoLlamadoAnalisisDeTexto = analizar_texto_basico(todasLasPalabrasJuntasTokenizadas)

        cantidadPalabras = f'Cantidad de palabras tokenizadas extraidas de todos los parafos del sitio web {url}: {len(todasLasPalabrasJuntasTokenizadas)}'

        textoARetornar = resultadoLlamadoAnalisisDeTexto + "\n\n" +cantidadPalabras+ "\n\n" + todasLasPalabrasJuntas

        return textoARetornar

    except Exception as e:
        print("ocurrió algo inesperado durante la ejecución")
        return f"se presentó el siguiente error:\n\n {e}"

def analizar_texto_basico(tokens):
    
    # Filtrar stopwords y puntuación
    stop_words = set(stopwords.words('spanish') + list(string.punctuation))
    palabras_filtradas = [palabra for palabra in tokens if palabra not in stop_words and palabra.isalpha()]

    print("stop_words:\n")
    print(stop_words)
    
    # Contar frecuencia
    fdist = FreqDist(palabras_filtradas)
    
    # Mostrar las 10 palabras más comunes
    print("10 palabras más frecuentes:")
    resultadoAnalisisDeTexto = "10 palabras más frecuentes:\n\n"

    for palabra, frecuencia in fdist.most_common(10):
        print(f"'{palabra}': {frecuencia} veces")
        resultadoAnalisisDeTexto = resultadoAnalisisDeTexto + "\n" + f"'{palabra}': {frecuencia} veces\n"
    
    return resultadoAnalisisDeTexto


st.title('Consumir sitio web y contar palabras')
valorTexto = st.text_input("Ingrese URL a consumir", "")






if st.button('Consumir Sitio Web'):
        #st.success(f'El resultado de la suma es: **{suma}**')
        st.write(f'Consumiendo el sitio web: ... {valorTexto}')

        resultado = consumirURL(valorTexto)

        st.write("---")
        st.write("---")
        st.write(resultado)