 Versão 1.1 - MVP do APP - Upload de Imagem (Desktop) -> Conversão para Fluxograma -> Análise Fuzzy

import streamlit as st
import cv2
import pytesseract
from PIL import Image
import graphviz
import numpy as np
from skfuzzy import control as ctrl
import tempfile

# --- MÓDULOS BASE ---

# Função: Pré-processamento da imagem
def preprocess_image(uploaded_image):
    image = np.array(Image.open(uploaded_image).convert('RGB'))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
    return thresh

# Função: OCR e Extração de texto
def extract_text(image):
    text = pytesseract.image_to_string(image)
    return text

# Função: Geração de Fluxograma básico
def generate_diagram(text):
    dot = graphviz.Digraph()
    lines = text.split('\n')
    for idx, line in enumerate(lines):
        if line.strip() != '':
            dot.node(f'P{idx}', line.strip())
            if idx > 0:
                dot.edge(f'P{idx-1}', f'P{idx}')
    return dot

# Função: Análise Fuzzy simples
def fuzzy_analysis(num_processes, has_loop):
    process = ctrl.Antecedent(np.arange(0, 20, 1), 'process')
    loop = ctrl.Antecedent(np.arange(0, 2, 1), 'loop')
    risk = ctrl.Consequent(np.arange(0, 101, 1), 'risk')

    process['few'] = ctrl.trimf(process.universe, [0, 0, 10])
    process['many'] = ctrl.trimf(process.universe, [5, 15, 20])
    loop['no'] = ctrl.trimf(loop.universe, [0, 0, 1])
    loop['yes'] = ctrl.trimf(loop.universe, [0, 1, 1])

    risk['low'] = ctrl.trimf(risk.universe, [0, 0, 50])
    risk['high'] = ctrl.trimf(risk.universe, [50, 100, 100])

    rule1 = ctrl.Rule(process['few'] & loop['no'], risk['low'])
    rule2 = ctrl.Rule(process['many'] | loop['yes'], risk['high'])

    risk_ctrl = ctrl.ControlSystem([rule1, rule2])
    risk_sim = ctrl.ControlSystemSimulation(risk_ctrl)

    risk_sim.input['process'] = num_processes
    risk_sim.input['loop'] = 1 if has_loop else 0

    risk_sim.compute()
    return risk_sim.output['risk']

# --- INTERFACE STREAMLIT ---
st.title("Conversor de Desenho para Fluxograma + Análise Fuzzy (Desktop)")

# Dropdown para escolha
input_option = st.selectbox("Selecione o método de entrada da imagem:", ["Upload de Arquivo", "Usar Câmera Desktop"])

uploaded_file = None

if input_option == "Upload de Arquivo":
    uploaded_file = st.file_uploader("Faça upload de uma imagem do seu desenho", type=['png', 'jpg', 'jpeg'])
elif input_option == "Usar Câmera Desktop":
    uploaded_file = st.camera_input("Capture uma imagem do seu desenho")

if uploaded_file:
    st.image(uploaded_file, caption='Imagem Original', use_column_width=True)

    st.subheader("Pré-processamento da Imagem")
    processed = preprocess_image(uploaded_file)
    st.image(processed, caption='Imagem Processada (Binarizada)')

    st.subheader("Texto Detectado (OCR)")
    extracted_text = extract_text(processed)
    st.text(extracted_text)

    st.subheader("Fluxograma Gerado")
    diagram = generate_diagram(extracted_text)
    st.graphviz_chart(diagram)

    st.subheader("Análise Fuzzy do Fluxo")
    num_steps = len([line for line in extracted_text.split('\n') if line.strip() != ''])
    has_loop = st.checkbox("Identificou loops ou retornos no desenho?")
    risk_score = fuzzy_analysis(num_steps, has_loop)
    st.metric("Risco de Falha/Complexidade (%)", f"{risk_score:.1f}%")

    st.success("Análise concluída! Sugestões detalhadas podem ser adicionadas na próxima versão.")

st.sidebar.title("Configurações")
st.sidebar.write("Todas as bibliotecas utilizadas possuem Pertinência Crítica (Peso 1.0) para execução no Streamlit.app.")

st.sidebar.markdown("---")
st.sidebar.write("Versão 1.1 - Preparada para futura integração com Google Apps e deploy mobile.")
