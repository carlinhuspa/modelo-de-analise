import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Adicionar o diretório atual ao path
sys.path.append(os.path.dirname(__file__))

# Importar os módulos de análise de texto
from analisador_texto import AnalisadorTextoEstatistico
from modelos_estatisticos import ModelosEstatisticos

def main():
    # Configuração da página
    st.set_page_config(
        page_title="Analisador de Estatísticas de Futebol",
        page_icon="⚽",
        layout="wide"
    )
    
    st.title("Analisador de Estatísticas de Futebol")
    
    # Adicionar CSS personalizado
    st.markdown("""
    <style>
    .main-header {
        color: #1E88E5;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .section-header {
        color: #1E88E5;
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #E3F2FD;
        border-left: 5px solid #1E88E5;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FFF8E1;
        border-left: 5px solid #FFC107;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .analysis-box {
        background-color: #F5F5F5;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar para navegação
    st.sidebar.title("Navegação")
    pagina = st.sidebar.radio("Escolha uma opção:", ["Análise Estatística", "Análise Científica", "Sobre"])
    
    if pagina == "Análise Estatística":
        st.header("Análise Estatística de Futebol")
        
        st.markdown("""
        <div class="info-box">
        Cole o texto estatístico do FootyStats ou similar para obter uma análise detalhada.
        </div>
        """, unsafe_allow_html=True)
        
        # Carregar texto estatístico
        texto_estatistico = st.text_area(
            "Cole o texto estatístico aqui (formato FootyStats):",
            height=300
        )
        
        if st.button("Analisar Estatísticas"):
            if not texto_estatistico:
                st.error("Por favor, forneça o texto estatístico.")
            else:
                with st.spinner("Analisando texto estatístico..."):
                    # Criar analisador
                    analisador = AnalisadorTextoEstatistico()
                    
                    # Analisar texto estatístico
                    analisador.carregar_texto(texto_estatistico)
                    
                    # Calcular probabilidades
                    analisador.calcular_probabilidades_poisson()
                    
                    # Gerar análise completa
                    analise = analisador.gerar_analise_completa()
                    
                    # Gerar texto de análise
                    analise_texto = analisador.gerar_texto_analise_estatistica()
                    
                    # Mostrar análise
                    st.subheader("Análise Estatística Detalhada")
                    st.markdown(f"""
                    <div class="analysis-box">
                    {analise_texto.replace('\n', '<br>')}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Mostrar visualizações
                    st.subheader("Visualizações")
                    
                    # Matriz de Poisson
                    st.write("**Matriz de Probabilidades (Poisson)**")
                    fig = analisador.plotar_matriz_poisson()
                    st.pyplot(fig)
                    
                    # Probabilidades de resultados
                    st.write("**Probabilidades de Resultados**")
                    fig = analisador.plotar_probabilidades_resultados()
                    st.pyplot(fig)
    
    elif pagina == "Análise Científica":
        st.header("Análise Científica com Poisson, xG e Odds")
        
        st.markdown("""
        <div class="info-box">
        Cole o texto estatístico do FootyStats e o texto de análise científica para obter uma análise completa.
        </div>
        """, unsafe_allow_html=True)
        
        # Carregar texto estatístico
        texto_estatistico = st.text_area(
            "Cole o texto estatístico aqui (formato FootyStats):",
            height=200
        )
        
        # Carregar texto científico
        texto_cientifico = st.text_area(
            "Cole o texto de análise científica aqui (Poisson, xG, etc.):",
            height=200
        )
        
        if st.button("Gerar Análise Científica"):
            if not texto_estatistico:
                st.error("Por favor, forneça o texto estatístico.")
            else:
                with st.spinner("Gerando análise científica..."):
                    # Criar analisador
                    analisador = AnalisadorTextoEstatistico()
                    
                    # Analisar texto estatístico
                    analisador.carregar_texto(texto_estatistico)
                    
                    # Se tiver texto científico, analisar também
                    if texto_cientifico:
                        analisador.carregar_texto_cientifico(texto_cientifico)
                    
                    # Calcular probabilidades
                    analisador.calcular_probabilidades_poisson()
                    
                    # Gerar análise científica
                    analise_cientifica = analisador.gerar_texto_analise_cientifica()
                    
                    # Mostrar análise
                    st.subheader("Análise Científica Detalhada")
                    st.markdown(f"""
                    <div class="analysis-box">
                    {analise_cientifica.replace('\n', '<br>')}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Mostrar recomendações
                    recomendacoes = analisador.gerar_recomendacoes()
                    if recomendacoes:
                        st.subheader("Recomendações de Apostas")
                        for i, rec in enumerate(recomendacoes):
                            st.markdown(f"""
                            <div class="success-box">
                            <strong>Recomendação {i+1}:</strong> {rec}
                            </div>
                            """, unsafe_allow_html=True)
    
    else:  # Sobre
        st.header("Sobre o Analisador de Estatísticas de Futebol")
        
        st.markdown("""
        <div class="info-box">
        Este aplicativo analisa textos estatísticos de futebol para extrair informações relevantes e gerar análises detalhadas.
        </div>
        
        ### Funcionalidades:
        
        - **Análise Estatística**: Extrai dados de textos estatísticos (como do FootyStats) e gera análises detalhadas.
        - **Análise Científica**: Aplica modelos matemáticos como Poisson e xG para gerar análises científicas.
        - **Visualizações**: Gera visualizações como matriz de probabilidades e gráficos de resultados.
        - **Recomendações**: Sugere apostas com base nas análises realizadas.
        
        ### Como usar:
        
        1. Navegue até a página "Análise Estatística" ou "Análise Científica"
        2. Cole o texto estatístico na área de texto
        3. Se estiver na página "Análise Científica", cole também o texto de análise científica
        4. Clique no botão para gerar a análise
        5. Visualize os resultados e recomendações
        
        ### Tecnologias utilizadas:
        
        - Python
        - Streamlit
        - Pandas
        - NumPy
        - Matplotlib
        - Seaborn
        - SciPy (para distribuição de Poisson)
        
        ### Desenvolvido por:
        
        Manus AI - 2025
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
