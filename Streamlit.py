import streamlit as st

st.title('Calculadora Simple con Streamlit')

valor1 = st.number_input('Introduce el primer valor:', min_value=0.0, value=10.0)

valor2 = st.number_input('Introduce el segundo valor:', min_value=0.0, value=5.0)

st.write('---') # Separador visual

suma = valor1 + valor2

if st.button('Calcular Suma'):
    st.success(f'El resultado de la suma es: *{suma}*')

st.subheader('Valores Ingresados')
st.write(f'Primer valor: *{valor1}*')
st.write(f'Segundo valor: *{valor2}*')