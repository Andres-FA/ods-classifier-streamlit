import streamlit as st
import joblib

st.set_page_config(page_title="Clasificador ODS", page_icon="🌍")

@st.cache_resource
def load_model():
    model = joblib.load("mejor_modelo_final.pkl")
    return model

model = load_model()

st.title("Clasificador de Objetivos de Desarrollo Sostenible")
st.write("Ingrese un texto y el modelo predecirá el ODS correspondiente.")

texto = st.text_area("Texto a clasificar")

if st.button("Predecir ODS"):

    if texto.strip() == "":
        st.warning("Por favor ingrese un texto")
    else:
        pred = model.predict([texto])[0]

        st.success(f"ODS Predicho: {pred}")
