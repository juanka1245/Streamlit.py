
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from openai import OpenAI
from openai import APIError, AuthenticationError


st.set_page_config(
    page_title="Prediccion de creditos",
    layout="wide"
)

st.title("Prediccion de creditos")



DATASET_PATH = "dataset_ml.csv"

if not os.path.exists(DATASET_PATH):
    np.random.seed(42)
    data = {
        "edad": np.random.randint(18, 65, 500),
        "ingresos": np.random.randint(1000, 10000, 500),
        "puntaje_credito": np.random.randint(300, 850, 500),
        "deuda": np.random.randint(0, 5000, 500),
        "aprobado": np.random.randint(0, 2, 500)
    }
    pd.DataFrame(data).to_csv(DATASET_PATH, index=False)



df = pd.read_csv(DATASET_PATH)

st.subheader("Dataset cargado")
st.dataframe(df.head())



st.subheader("Visualización de Datos")

fig, ax = plt.subplots()
ax.hist(df["puntaje_credito"], bins=30)
ax.set_title("Distribución del Puntaje de Crédito")
st.pyplot(fig)



X = df.drop("aprobado", axis=1)
y = df["aprobado"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)



st.subheader("Resultados del Modelo")

st.write(f"**Accuracy:** {accuracy:.2f}")

st.text("Reporte de Clasificación:")
st.text(classification_report(y_test, y_pred))



st.subheader("🧪 Predicción con Datos Nuevos")

edad = st.number_input("Edad", 18, 100, 30)
ingresos = st.number_input("Ingresos", 500, 20000, 3000)
puntaje = st.number_input("Puntaje de Crédito", 300, 900, 650)
deuda = st.number_input("Deuda", 0, 10000, 1000)

if st.button("Predecir"):
    nuevo = scaler.transform([[edad, ingresos, puntaje, deuda]])
    resultado = model.predict(nuevo)[0]

    if resultado == 1:
        st.success("✅ Crédito APROBADO")
    else:
        st.error("❌ Crédito NO aprobado")



st.subheader("🤖 Análisis con OpenAI (opcional)")

consulta = st.text_area(
    "Pregunta sobre el modelo o los datos:",
    "Explica el desempeño del modelo"
)

if st.button("Consultar IA"):
    try:
     
        client = OpenAI(api_key="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")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Eres un experto en machine learning."},
                {"role": "user", "content": consulta}
            ]
        )

        st.write(response.choices[0].message.content)

    except AuthenticationError:
        st.error("❌ Error de autenticación: revisa tu API Key.")

    except APIError as e:
        st.error(f"❌ Error de la API de OpenAI: {e}")

    except Exception as e:
        st.error(f"❌ Error inesperado: {e}")


st.markdown("---")
st.header("💬 Chatbot Interactivo ")
st.caption("Realizame una consulta.")


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hola! Soy tu asistente. Realizame una consulta."}
    ]

try:
    
    openai_client = OpenAI(api_key="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")
   
    is_client_ready = True
except Exception:
   
    st.warning("⚠️ La API Key de OpenAI no está configurada (OPENAI_API_KEY). El chatbot no funcionará.")
    openai_client = None
    is_client_ready = False


if not is_client_ready:
    # Este bloque solo se ejecutará si la clave NO funciona o la pusiste mal.
    # Dado que ya pusiste la clave en las líneas 204 y 240, este bloque no debería ejecutarse.
    st.warning("La API Key de OpenAI no está configurada (OPENAI_API_KEY). El chatbot no funcionará.")


# 3. Mostrar mensajes anteriores
for msg in st.session_state.messages:
    # Usamos st.chat_message para la interfaz de chat de Streamlit
    st.chat_message(msg["role"]).write(msg["content"])

# 4. Manejar la entrada del usuario
if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    # Añadir el mensaje del usuario al historial
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if openai_client:
        try:
            # Llamar a la API de OpenAI con el historial completo
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                # Se envía la lista completa de mensajes para mantener el contexto
                messages=st.session_state.messages
            )
            
            # Obtener la respuesta del asistente
            msg = response.choices[0].message.content
            
            # Mostrar la respuesta y añadirla al historial
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)

        except AuthenticationError:
            error_msg = "Error de autenticación: revisa tu API Key."
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.chat_message("assistant").write(error_msg)

        except APIError as e:
            error_msg = f"Error de la API de OpenAI: {e}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.chat_message("assistant").write(error_msg)

        except Exception as e:
            error_msg = f"Error inesperado: {e}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.chat_message("assistant").write(error_msg)
    else:
        # Mensaje si el cliente OpenAI no está inicializado
        st.chat_message("assistant").write("No puedo responder, la API Key de OpenAI no está configurada correctamente.")

# Botón para limpiar el historial de chat
if st.button("Limpiar Chat"):
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Historial limpiado. ¡Hola de nuevo!"}
    ]
    st.experimental_rerun() # Recargar para reflejar el cambio



st.markdown("---")
st.caption("Proyecto ML Avanzado • Pandas • Matplotlib • OpenAI • Streamlit")
