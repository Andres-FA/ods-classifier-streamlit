import joblib
import streamlit as st

# =========================
# Configuración de página
# =========================
st.set_page_config(
    page_title="Clasificador ODS",
    page_icon="🌍",
    layout="centered"
)

# =========================
# Carga del modelo
# =========================
@st.cache_resource
def load_model():
    return joblib.load("mejor_modelo_final.pkl")

model = load_model()

# =========================
# Diccionario opcional de nombres ODS
# =========================
ODS_NAMES = {
    1: "ODS 1 - Fin de la pobreza",
    2: "ODS 2 - Hambre cero",
    3: "ODS 3 - Salud y bienestar",
    4: "ODS 4 - Educación de calidad",
    5: "ODS 5 - Igualdad de género",
    6: "ODS 6 - Agua limpia y saneamiento",
    7: "ODS 7 - Energía asequible y no contaminante",
    8: "ODS 8 - Trabajo decente y crecimiento económico",
    9: "ODS 9 - Industria, innovación e infraestructura",
    10: "ODS 10 - Reducción de las desigualdades",
    11: "ODS 11 - Ciudades y comunidades sostenibles",
    12: "ODS 12 - Producción y consumo responsables",
    13: "ODS 13 - Acción por el clima",
    14: "ODS 14 - Vida submarina",
    15: "ODS 15 - Vida de ecosistemas terrestres",
    16: "ODS 16 - Paz, justicia e instituciones sólidas",
    17: "ODS 17 - Alianzas para lograr los objetivos"
}

# =========================
# Estado inicial
# =========================
if "texto_usuario" not in st.session_state:
    st.session_state["texto_usuario"] = ""

# =========================
# Interfaz
# =========================
st.title("🌍 Clasificador de textos por ODS")
st.write(
    "Ingrese un texto libre. La aplicación procesará el contenido con el "
    "mismo pipeline entrenado en el proyecto y mostrará el ODS predicho."
)

texto = st.text_area(
    "Texto de entrada",
    height=220,
    placeholder="Escriba aquí el texto que desea clasificar...",
    key="texto_usuario"
)

col1, col2 = st.columns(2)

with col1:
    predecir = st.button("Predecir ODS", use_container_width=True)

with col2:
    limpiar = st.button("Limpiar", use_container_width=True)

if limpiar:
    st.session_state["texto_usuario"] = ""
    st.rerun()

if predecir:
    if not st.session_state["texto_usuario"].strip():
        st.warning("Por favor, ingrese un texto antes de ejecutar la predicción.")
    else:
        try:
            pred = model.predict([st.session_state["texto_usuario"]])[0]

            # Convertir a int por seguridad
            try:
                pred_num = int(pred)
            except:
                pred_num = pred

            st.success(f"ODS predicho: {pred_num}")

            if pred_num in ODS_NAMES:
                st.info(f"Descripción: {ODS_NAMES[pred_num]}")

            st.subheader("Texto evaluado")
            st.write(st.session_state["texto_usuario"])

        except Exception as e:
            st.error("Ocurrió un error al procesar el texto.")
            st.exception(e)

st.markdown("---")
st.caption("Aplicación desarrollada en Streamlit para clasificación automática de textos por ODS.")
