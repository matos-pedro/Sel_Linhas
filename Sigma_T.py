import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spec_utils import analisar_sigma_T
from matplotlib.colors import LogNorm

st.set_page_config(layout='wide')
st.header("Seleção de Linhas")  
st.subheader("Análise de Temperatura via Perturbações nas Intensidades de Linha")
# === Sidebar: parâmetros de entrada ===
st.sidebar.title("⚙️ Parâmetros de Entrada")

T = st.sidebar.number_input("Temperatura Simulada T (K)", value=1500, step=1)
range_1_nm = st.sidebar.text_input("Região espectral 1 (nm)", "1342.937, 1345.779")
range_2_nm = st.sidebar.text_input("Região espectral 2 (nm)", "1390.13, 1393.17")

try:
    range_1 = tuple(map(float, range_1_nm.split(',')))
    range_2 = tuple(map(float, range_2_nm.split(',')))
except:
    st.error("Erro ao interpretar os intervalos espectrais. Use vírgulas: ex. 1300,1310")
    st.stop()

mol_id = 1  # fixo, pode ajustar se quiser input
iso_id = 1
T_ref = 298
Nl = st.sidebar.number_input("Número de Linhas mais Intensas, NL", value=10, step=1)
K = st.sidebar.number_input("Número de Perturbações, NP", value=100, step=1)
erro_rel_I = st.sidebar.number_input("Erro relativo nas intensidades (%), ER", value=10, step=1) / 100.0

executar = st.sidebar.button("🚀 Executar análise")

# === Corpo principal com abas ===

aba_saida, aba_explicacao = st.tabs(["📊 Saída", "ℹ️ Teoria"])

with aba_saida:
    # Container para limitar a largura do conteúdo na aba saída
    with st.container():
        st.markdown(
            """
            <style>
            div.block-container {
                max-width: 1000px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        if executar:
            with st.spinner("Calculando..."):
                sigma_T_matrix, nu_1, nu_2 = analisar_sigma_T(
                    T=T,
                    range_1=range_1,
                    range_2=range_2,
                    mol_id=mol_id,
                    iso_id=iso_id,
                    T_ref=T_ref,
                    Nl=Nl,
                    nm=True,
                    erro_rel_I=erro_rel_I,
                    K=K
                )
            st.success("Análise finalizada!")

            st.markdown(f"""
                #### Parâmetros da análise
                - Temperatura simulada T: **{T} K**  
                - Região espectral 1 (nm): **{range_1[0]} a {range_1[1]}**  
                - Região espectral 2 (nm): **{range_2[0]} a {range_2[1]}**  
                - Número de linhas mais intensas: **{Nl}**  
                - Número de perturbações K: **{K}**  
                - Erro relativo nas intensidades: **{erro_rel_I*100:.1f}%**
                
                O gráfico abaixo mostra a matriz de σ(T) estimada com base nesses parâmetros. Cores mais claras indicam menor incerteza na temperatura.
                """)

            # Gráfico matplotlib
            fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
            sigma_plot = np.where(sigma_T_matrix > 0, sigma_T_matrix, np.nan)
            im = ax.imshow(
                sigma_plot,
                cmap='coolwarm',
                aspect='auto',
                norm=LogNorm(vmin=np.nanmin(sigma_plot), vmax=np.nanmax(sigma_plot))
            )
            ax.set_title(f"σ(T) a T = {T} K")
            ax.set_xlabel("Região 2 (nm)")
            ax.set_ylabel("Região 1 (nm)")

            ax.set_xticks(np.arange(len(nu_2)))
            ax.set_yticks(np.arange(len(nu_1)))
            ax.set_xticklabels([f"{1e7 / n:.2f}" for n in nu_2], rotation=45, ha='right')
            ax.set_yticklabels([f"{1e7 / n:.2f}" for n in nu_1])
            plt.colorbar(im, ax=ax, label="σ(T) [K]")
            plt.tight_layout()

            st.pyplot(fig)

            # Tabela dentro de expander
            with st.expander("📋 Mostrar Matriz σ(T)"):
                col_labels = [f"{1e7 / n:.4f}" for n in nu_2]
                row_labels = [f"{1e7 / n:.4f}" for n in nu_1]
                df_sigma = pd.DataFrame(np.round(sigma_T_matrix, 2), index=row_labels, columns=col_labels)

                st.markdown("Linhas = Região 1 (nm) &nbsp;&nbsp;&nbsp;&nbsp; Colunas = Região 2 (nm)", unsafe_allow_html=True)
                st.dataframe(df_sigma.style.set_table_styles(
                    [{"selector": "th", "props": [("font-weight", "bold")]}]
                ), use_container_width=True)
        else:
            st.info("Use a barra lateral para configurar os parâmetros e clique em '🚀 Executar análise'.")

with aba_explicacao:
    st.markdown("""
    #### Estimativas Espectroscópicas da Água usando o HITRAN

    Este documento descreve três procedimentos importantes para espectroscopia baseada na base de dados HITRAN:

    1. Como estimar a força de linha $S(T)$ a uma temperatura arbitrária,  
    2. Como estimar a temperatura a partir da razão entre duas linhas espectrais e
    3. Como o código funciona. 

    ---

    ### 1. Estimando a Força da Linha a uma Temperatura Arbitrária

    A força de linha espectral $S(T)$ depende fortemente da temperatura. A base de dados HITRAN fornece essa força $S_{\\text{ref}}$ para uma temperatura de referência $T_{\\text{ref}} = 296\,\\text{K}$, mas para simulações em outras temperaturas, é necessário corrigi-la.

    A correção é dada por:

    $$
    S(T) = S(T_{\\text{ref}}) \\cdot \\frac{Q(T_{\\text{ref}})}{Q(T)} \\cdot \\exp\\left(-\\frac{c_2 E''}{T} + \\frac{c_2 E''}{T_{\\text{ref}}} \\right) \\cdot \\frac{1 - \\exp\\left(-\\frac{c_2 \\nu}{T} \\right)}{1 - \\exp\\left(-\\frac{c_2 \\nu}{T_{\\text{ref}}} \\right)}
    $$

    onde:  
    - $S(T_{\\text{ref}})$ é a força fornecida pelo HITRAN a 296 K,  
    - $Q(T)$ é a função partição em temperatura $T$, representando a distribuição estatística dos estados energéticos,  
    - $\\nu$ é o número de onda da transição (em cm⁻¹),  
    - $E''$ é a energia do estado inferior (em cm⁻¹),  
    - $c_2 = \\frac{hc}{k_B} \\approx 1.4388\,\\text{cm}\\cdot\\text{K}$ é a constante espectroscópica,  
    - O último fator corrige para o estímulo térmico na população dos estados.

    Essa fórmula leva em conta a variação da população dos níveis vibracionais e rotacionais com a temperatura, bem como o efeito do estímulo térmico na transição.

    ---

    ### 2. Estimando a Temperatura a partir de Duas Linhas

    A razão entre as intensidades de duas linhas espectrais pode ser usada para estimar a temperatura do meio, aproveitando que cada linha tem uma dependência diferente da temperatura devido às diferentes energias dos níveis inferiores.

    Dado que as intensidades medidas são $I_1$ e $I_2$, e as forças de linha de referência são $S_1(T_{\\text{ref}})$ e $S_2(T_{\\text{ref}})$, a temperatura $T$ pode ser estimada iterativamente pela equação:

    $$
    T = -\\frac{c_2 (E_1 - E_2)}{\\ln R}
    $$

    onde

    $$
    R = \\frac{I_1}{I_2} \\cdot \\frac{S_2(T_{\\text{ref}})}{S_1(T_{\\text{ref}})} \\cdot \\frac{F_2(T)}{F_1(T)} \\cdot \\exp\\left(-\\frac{c_2 (E_1 - E_2)}{T_{\\text{ref}}} \\right)
    $$

    e

    $$
    F(T) = \\frac{1 - \\exp\\left(-\\frac{c_2 \\nu}{T} \\right)}{1 - \\exp\\left(-\\frac{c_2 \\nu}{T_{\\text{ref}}} \\right)}
    $$

    é o fator de correção térmica para cada linha.

    Esse método assume que as condições experimentais (pressão, caminho óptico, concentração) são as mesmas para ambas as linhas, para que essas variáveis se cancelem na razão.

    O cálculo de $T$ pode ser feito iterativamente, pois $F(T)$ depende da temperatura que está sendo estimada. Em muitos casos, poucas iterações são suficientes para convergência.

    ##### 2.1 Calculando a Intensidade de Absorção




    ---

     ### 3. Seleção de Linhas e Estimativa da Incerteza na Temperatura

     O código calcula uma matriz de desvio padrão $\sigma(T)$ para a temperatura estimada, simulando o impacto do ruído experimental nas intensidades de absorção das linhas espectrais.

     Para isso, são considerados pares de linhas, onde uma linha é escolhida de uma região espectral e a outra linha é escolhida da segunda região especificada pelo usuário. Essa combinação cruzada permite analisar a sensibilidade da estimativa da temperatura a partir de diferentes pares.

     Para cada par de linhas:

     - A intensidade de cada linha é perturbada diversas vezes com ruído gaussiano, cuja amplitude é proporcional ao maior valor de intensidade encontrado entre todas as linhas das duas regiões, multiplicado pelo erro relativo definido pelo usuário.
     - A temperatura é estimada para cada par perturbado, utilizando o método iterativo baseado na razão das intensidades.
     - Calcula-se o desvio padrão das temperaturas estimadas em relação à temperatura simulada original, resultando na incerteza $\sigma(T)$ para aquele par.

     A matriz resultante de $\sigma(T)$ é apresentada em um gráfico de calor, onde cores mais claras indicam pares de linhas que proporcionam estimativas de temperatura mais estáveis e confiáveis (menor incerteza). Isso ajuda a selecionar os pares de linhas mais robustos para diagnósticos espectroscópicos e para determinar a temperatura com maior precisão.

     

     ###### Explicação dos parâmetros de entrada

     - **Temperatura Simulada T (K):** Temperatura real do sistema que se deseja estimar via espectroscopia.
     - **Região espectral 1 (nm):** Intervalo da primeira região espectral, onde serão selecionadas linhas para a análise.
     - **Região espectral 2 (nm):** Intervalo da segunda região espectral para seleção de linhas.
     - **Número de Linhas mais Intensas, NL:** Quantidade de linhas mais fortes a serem consideradas em cada região para formar os pares.
     - **Número de Perturbações, K:** Quantidade de amostras de ruído geradas para simular a incerteza nas intensidades.
     - **Erro relativo nas intensidades, ER:** Porcentagem do maior valor de intensidade usada para definir o desvio padrão do ruído gaussiano aplicado às intensidades.

     Esses parâmetros permitem ajustar a análise para o caso experimental, controlando a quantidade de linhas consideradas, o nível de ruído simulado e as regiões espectrais de interesse.
     
     ---

    ### Considerações Finais

    - Aqui, foram desconsiderados os efeitos de alargamento e as linhas são, portanto, isoladas. Experimentalmente, as linhas são afetadas por muitos fatores ambientais e instrumentais.
    - A incerteza na estimativa da temperatura é avaliada por meio de simulações com ruído gaussiano adicionado às intensidades espectrais, permitindo quantificar o impacto de flutuações experimentais sobre a robustez dos pares de linhas selecionados   
    - A base HITRAN fornece dados essenciais para simulações e análise espectral.
                
    ### Referência
    
    GRIFFITHS, A. D. Development and Demonstration of a Diode Laser Sensor for a Scramjet Combustor. 2005. Tese de Doutorado. Thesis (Doctor of Philosophy)–The Australian Natonal University, Australia.
    [Disponível aqui](https://www.researchgate.net/profile/Alan-Griffiths/publication/265320975_Development_and_demonstration_of_a_diode_laser_sensor_for_a_scramjet_combustor/links/58b35a2c92851cf7ae91d5d1/Development-and-demonstration-of-a-diode-laser-sensor-for-a-scramjet-combustor.pdf)
    """)