# ==============================
# IMPORTACIONES
# ==============================
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

# ==============================
# CONFIGURACIÓN STREAMLIT
# ==============================
st.set_page_config(
    page_title="ML Avanzado con Streamlit",
    layout="wide"
)

st.title("🚀 Proyecto Avanzado de Machine Learning en Python")


# ==============================
# CREAR DATASET CSV (SI NO EXISTE)
# ==============================
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


# ==============================
# CARGA DE DATOS
# ==============================
df = pd.read_csv(DATASET_PATH)

st.subheader("📊 Dataset cargado")
st.dataframe(df.head())


# ==============================
# VISUALIZACIÓN
# ==============================
st.subheader("📈 Visualización de Datos")

fig, ax = plt.subplots()
ax.hist(df["puntaje_credito"], bins=30)
ax.set_title("Distribución del Puntaje de Crédito")
st.pyplot(fig)


# ==============================
# PREPROCESAMIENTO
# ==============================
X = df.drop("aprobado", axis=1)
y = df["aprobado"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# ==============================
# MODELO ML AVANZADO
# ==============================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)


# ==============================
# RESULTADOS
# ==============================
st.subheader("🧠 Resultados del Modelo")

st.write(f"**Accuracy:** {accuracy:.2f}")

st.text("Reporte de Clasificación:")
st.text(classification_report(y_test, y_pred))


# ==============================
# RUTAS DE ENTRADA (INPUT USUARIO)
# ==============================
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


# ==============================
# INTEGRACIÓN OPENAI (MANTENIDA)
# ==============================
st.subheader("🤖 Análisis con OpenAI (opcional)")

consulta = st.text_area(
    "Pregunta sobre el modelo o los datos:",
    "Explica el desempeño del modelo"
)

if st.button("Consultar IA"):
    try:
        # CORRECCIÓN APLICADA AQUÍ: Se usa la clave directamente.
        client = OpenAI(api_key="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")

        # CAMBIO 1: Sistema de instrucción modificado para enfocar el análisis
        SYSTEM_PROMPT_ANALISIS = "Eres un experto en Machine Learning que solo responde preguntas relacionadas con el código de Streamlit, el modelo RandomForest, el dataset (edad, ingresos, puntaje_credito, deuda, aprobado) y los resultados de este proyecto específico. Si la pregunta no es relevante al proyecto o al código, debes indicarlo."

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_ANALISIS},
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

# --- SECCIÓN DE CHATBOT INTERACTIVO AÑADIDA ---
st.markdown("---")
st.header("💬 Chatbot Interactivo de ML")
st.caption("Pregunta lo que quieras sobre el modelo, los datos, o conceptos generales de Machine Learning.")

# 1. Inicializar el historial del chat
SYSTEM_PROMPT_CHATBOT = "Eres un asistente de Machine Learning especializado en el código de Streamlit, el modelo RandomForest, los datos (edad, ingresos, puntaje_credito, deuda, aprobado) y los resultados de este proyecto de predicción de crédito. Tu única función es ayudar al usuario a entender el código y el proyecto. Si el usuario pregunta algo fuera de este contexto, responde amablemente que solo puedes hablar sobre el proyecto actual."

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        # La primera respuesta del asistente también refleja la nueva limitación
        {"role": "assistant", "content": "Hola! Soy tu asistente especializado. Pregúntame sobre el código, el dataset (edad, ingresos, puntaje de crédito, etc.), el modelo RandomForest o el rendimiento de este proyecto de predicción de crédito."}
    ]

# 2. Conectar el cliente de OpenAI
try:
    # CORRECCIÓN APLICADA AQUÍ: Se usa la clave directamente.
    openai_client = OpenAI(api_key="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")
    is_client_ready = True
except Exception:
    openai_client = None
    is_client_ready = False

# HACK PARA EVITAR EL MENSAJE DE ERROR QUE APARECE AL PRINCIPIO
if not is_client_ready:
    st.warning("⚠️ La API Key de OpenAI no está configurada (OPENAI_API_KEY). El chatbot no funcionará.")


# 3. Mostrar mensajes anteriores
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 4. Manejar la entrada del usuario
if prompt := st.chat_input("Escribe tu pregunta aquí..."):
    
    # 4a. Preparar mensajes con la instrucción de sistema
    messages_with_system_prompt = [
        {"role": "system", "content": SYSTEM_PROMPT_CHATBOT}
    ]
    # Añadir el historial de la conversación (usuario/asistente)
    messages_with_system_prompt.extend(st.session_state.messages)

    # Añadir el mensaje del usuario al historial para visualización
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if openai_client:
        try:
            # Llamar a la API de OpenAI con el historial y la nueva restricción de contexto
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                # CAMBIO 2: Se envía la lista con el SYSTEM_PROMPT al inicio
                messages=messages_with_system_prompt
            )
            
            msg = response.choices[0].message.content
            
            # Mostrar la respuesta y añadirla al historial
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)

        except AuthenticationError:
            error_msg = "❌ Error de autenticación: revisa tu API Key."
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.chat_message("assistant").write(error_msg)

        except APIError as e:
            error_msg = f"❌ Error de la API de OpenAI: {e}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.chat_message("assistant").write(error_msg)

        except Exception as e:
            error_msg = f"❌ Error inesperado: {e}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.chat_message("assistant").write(error_msg)
    else:
        st.chat_message("assistant").write("No puedo responder, la API Key de OpenAI no está configurada correctamente.")

# Botón para limpiar el historial de chat
if st.button("Limpiar Chat"):
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Historial limpiado. ¡Hola de nuevo! Recuerda, solo puedo hablar sobre el código y el proyecto de ML."}
    ]
    st.experimental_rerun() # Recargar para reflejar el cambio


# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.caption("Proyecto ML Avanzado • Pandas • Matplotlib • OpenAI • Streamlit")