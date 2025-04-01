import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc
import os

# URLs
linkedin_url = 'https://linkedin.com/comm/mynetwork/discovery-see-all?usecase=PEOPLE_FOLLOWS&followMember=brunoarandati'
icon_url = 'https://img.icons8.com/?size=100&id=8808&format=png&color=FFFFFF'
perfil_img_url = 'https://i.ibb.co/351BthXC/Black-Pattern-Minimalist-Linked-In-Profile-Picture.png'

def find_default_id_column(columns):
    id_keywords = ['id', 'usuario', 'user', 'client', 'customer']
    for col in columns:
        if any(keyword.lower() in col.lower() for keyword in id_keywords):
            return col
    return columns[0] if columns else None

def generate_synthetic_data():
    dataset_path = "datasets/synthetic_data.csv"
    
    # Cria a pasta 'datasets' se ela não existir
    os.makedirs("datasets", exist_ok=True)
    
    # Verifica se o arquivo já existe
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path, parse_dates=["date"])
        return df

    # Gerar os dados sintéticos
    np.random.seed(42)
    user_ids = np.arange(1, 501)
    start_dates = pd.date_range(start="2023-01-01", periods=12, freq="ME")
    data = []
    for user_id in user_ids:
        signup_date = np.random.choice(start_dates)
        source = np.random.choice(['organico', 'ads', 'indicacao'], p=[0.6, 0.3, 0.1])
        plan = np.random.choice(['free', 'premium'], p=[0.8, 0.2])
        for i in range(np.random.randint(1, 12)):
            activity_date = signup_date + pd.DateOffset(months=i)
            if activity_date > start_dates[-1]:
                break
            data.append((user_id, activity_date, source, plan))
    
    df = pd.DataFrame(data, columns=["user_id", "date", "source", "plan"])
    
    # Salvar no diretório datasets
    df.to_csv(dataset_path, index=False)
    
    return df

def is_valid_date_column(series, threshold=0.5):
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    try:
        converted = pd.to_datetime(series, errors='coerce')
        valid_count = converted.notna().sum()
        return (valid_count / len(series)) >= threshold
    except:
        return False

def get_valid_date_columns(df, exclude_columns=[], threshold=0.5):
    return [col for col in df.columns 
            if col not in exclude_columns 
            and is_valid_date_column(df[col], threshold)]

def find_default_date_column(columns):
    date_keywords = ['data', 'Data', 'date', 'Date']
    for col in columns:
        if col in date_keywords:
            return col
    return None

def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("✅ Dados carregados com sucesso.")
    else:
        df = generate_synthetic_data()
        st.sidebar.info("⚠️ Usando dados de exemplo.")
    return df

def preprocess_data(df, user_id_column, date_column, segment_column=None):
    df["cohort_month"] = df.groupby([user_id_column] + ([segment_column] if segment_column else []))[date_column].transform("min").dt.to_period("M").astype(str)
    df["order_month"] = df[date_column].dt.to_period("M").astype(str)

    all_months = sorted(pd.period_range(
        start=df["cohort_month"].min(),
        end=df["order_month"].max(),
        freq='M'
    ).astype(str))

    month_to_index = {month: idx for idx, month in enumerate(all_months)}

    df["cohort_index"] = (
        df["order_month"].map(month_to_index) -
        df["cohort_month"].map(month_to_index)
    )

    group_cols = ["cohort_month", "cohort_index"] + ([segment_column] if segment_column else [])
    cohort_data = df.groupby(group_cols).agg({user_id_column: "nunique"}).reset_index()

    if segment_column:
        cohort_pivot = cohort_data.pivot_table(
            index=["cohort_month", segment_column],
            columns="cohort_index",
            values=user_id_column,
            fill_value=0
        )
    else:
        cohort_pivot = cohort_data.pivot(
            index="cohort_month",
            columns="cohort_index",
            values=user_id_column
        )

    cohort_size = cohort_pivot.iloc[:, 0]
    retention = cohort_pivot.divide(cohort_size, axis=0) * 100
    retention = retention.fillna(0)

    return retention, cohort_pivot

def plot_interactive_cohort(retention, cohort_pivot, display_type, color_palette, segment_column=None):
    data_to_plot = retention if display_type == 'Percentual' else cohort_pivot
    z_data = data_to_plot.replace(0, np.nan)
    text_data = z_data.applymap(lambda x: 
                                "" if pd.isnull(x) else 
                                f"{x:.0f}%" if display_type == 'Percentual' else 
                                f"{x:.0f}")

    heatmap_args = {
        "colorscale": color_palette,
        "texttemplate": "%{text}",
        "textfont": {"size": 12},
        "hoverongaps": False,
        "xgap": 1,
        "ygap": 1,
        "showscale": False,
        "hovertemplate": "<b>Cohorte:</b> %{y}<br>"
                         + "<b>Mês:</b> %{x}<br>"
                         + "<b>Valor:</b> %{text}<extra></extra>"
    }

    if segment_column:
        segments = z_data.index.get_level_values(1).unique()
        fig = make_subplots(
            rows=len(segments),
            cols=1,
            subplot_titles=[f"Segmento: {seg}" for seg in segments],
            vertical_spacing=0.15
        )

        for i, segment in enumerate(segments, 1):
            segment_z = z_data.xs(segment, level=1)
            segment_text = text_data.xs(segment, level=1)

            heatmap = go.Heatmap(
                x=segment_z.columns,
                y=segment_z.index,
                z=segment_z.values,
                text=segment_text.values,
                **heatmap_args
            )

            fig.add_trace(heatmap, row=i, col=1)
            fig.update_xaxes(title_text="Meses desde a aquisição", row=i, col=1)
            fig.update_yaxes(title_text="Cohorte", row=i, col=1, autorange="reversed")

        shapes = []
        if len(segments) > 1:
            for i in range(1, len(segments)):
                y_pos = 1 - i / len(segments)
                shapes.append({
                    "type": "line",
                    "xref": "paper",
                    "yref": "paper",
                    "x0": 0,
                    "x1": 1,
                    "y0": y_pos,
                    "y1": y_pos,
                    "line": {
                        "color": "lightgray",
                        "width": 1,
                        "dash": "dash"
                    }
                })

        fig.update_layout(
            height=400 * len(segments),
            margin={"t": 40, "b": 20},
            showlegend=False,
            shapes=shapes
        )
    else:
        fig = go.Figure()
        heatmap = go.Heatmap(
            x=z_data.columns,
            y=z_data.index,
            z=z_data.values,
            text=text_data.values,
            **heatmap_args
        )
        fig.add_trace(heatmap)
        fig.update_layout(
            height=800,
            xaxis_title="Meses desde a aquisição",
            yaxis_title="Cohorte de aquisição",
            margin={"t": 40, "b": 20}
        )
        fig.update_yaxes(autorange="reversed")

    st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(page_title="Análise Cohort", layout="wide", page_icon="📈")
    
    # Sidebar
    with st.sidebar:
        st.markdown(
            f"""
            <div style='text-align: center;'>
                <img src='{perfil_img_url}' alt='Perfil' width='150' style='border-radius: 50%; margin-bottom: 10px;'>
                <h3>Siga-me no Linkedin 👇</h3>
                <a href='{linkedin_url}' target='_blank' style='text-decoration: none;'>
                    <button style='background-color:#0077B5; color:white; border:none; padding:10px 20px; border-radius:5px; font-size:16px; cursor:pointer; display:inline-flex; align-items:center;'>
                        <img src='{icon_url}' alt='LinkedIn' width='20' style='margin-right:8px;'> Siga-me
                    </button>
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("---")

        st.header("⚙️ Configurações")
        
        # Upload de dados
        uploaded_file = st.file_uploader("Carregar CSV", type=["csv"], help="O separador do CSV deve ser \",\" e o de casas decimais \".\"")
        df = load_data(uploaded_file)
        
        # Filtros
        st.subheader("Parâmetros")
        if not df.empty:
            # Seleção de ID do usuário
            valid_id_cols = [col for col in df.columns if df[col].notna().all() and df[col].nunique() > 1]
            # Encontrar coluna ID padrão automaticamente
            default_user_id_col = find_default_id_column(valid_id_cols)
            default_idx = valid_id_cols.index(default_user_id_col) if default_user_id_col in valid_id_cols else 0

            user_id_col = st.selectbox(
                "Coluna de identificação do usuário/cliente:",
                valid_id_cols,
                index=default_idx,
                help="Selecione a coluna com identificadores dos usuários"
            )
            
            # Validação e seleção de data
            valid_date_cols = get_valid_date_columns(df, exclude_columns=[user_id_col])
            default_date_col = find_default_date_column(valid_date_cols)
            
            if not valid_date_cols:
                st.error("⚠️ Nenhuma coluna de data válida encontrada!")
                st.stop()
                
            # Encontrar índice padrão
            default_idx = valid_date_cols.index(default_date_col) if default_date_col else 0
            
            date_col = st.selectbox(
                "Coluna de data do comportamento analisado:",
                valid_date_cols,
                index=default_idx,
                help="Colunas com pelo menos 50% de datas válidas"
            )

            # Conversão definitiva para datetime
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Segmentação
            cat_cols = [col for col in df.columns
                        if (df[col].nunique() < 15)
                        and (col not in [user_id_col, date_col])
                        and pd.api.types.is_string_dtype(df[col])]
            
            segment_col = None
            if cat_cols:
                segment_col = st.selectbox(
                    "Segmentar por:",
                    [None] + cat_cols,
                    format_func=lambda x: 'Nenhum' if x is None else x,
                    help="Ex: origem do lead, vededor responsável, onboarding aplicado"
                )
            
            # Filtro de datas
            min_date = df[date_col].min().date()
            max_date = df[date_col].max().date()
            selected_dates = st.date_input(
                "Intervalo de datas:",
                [min_date, max_date],
                min_value=min_date,
                max_value=max_date
            )
            
            # Aplicar filtro de datas
            df = df[(df[date_col].dt.date >= selected_dates[0]) & 
                    (df[date_col].dt.date <= selected_dates[1])]
            

            # Configurações de visualização
            st.subheader("Visualização")
            display_type = st.radio(
                "Tipo de exibição:",
                ["Percentual", "Quantidade"],
                horizontal=True,
                help="Mostrar valores absolutos ou porcentagem de retenção"
            )
            
            color_palettes = sorted([p for p in pc.sequential.__dict__ if not p.startswith('_')])
            color_palette = st.selectbox(
                "Paleta de cores:",
                color_palettes,
                index=color_palettes.index('Viridis'),
                help="Escolha um esquema de cores para o heatmap"
            )

        st.markdown("---")
        if st.button("❓ Ajuda Rápida"):
            st.info("""
            1. Carregue um CSV que tenha como caracter separador "," e decimal "."
            2. Ajuste os parâmetros selecionando a coluna de identificação do usuário e coluna data do evento analisado.
            3. Utilize as segmentações e as cores para adaptar a análise ao seu gosto.
            """)

    # Corpo principal
    st.title("📊 Análise de Retenção - Cohort")
    with st.expander("🤔 O que é a Análise Cohort?"):
        st.markdown("""
        A **Análise Cohort** agrupa usuários com características ou experiências comuns (por exemplo, data de cadastro, versão do produto utilizada ou campanha de marketing). Assim, você consegue acompanhar o comportamento desses "coortes" ao longo do tempo.

        **Que perguntas ela pode responder?**

        A Análise Cohort ajuda você a tomar decisões assertivas, respondendo questões críticas como:

        📌 Quando meus clientes estão abandonando o serviço?  
        📌 Qual versão do produto gera mais engajamento?  
        📌 Qual onboarding realmente aumenta a retenção?  
        📌 Quais campanhas trazem usuários que ficam mais tempo?  
        📌 Meu experimento recente teve o impacto esperado?

        Essas respostas afetam diretamente:

        📉 Sua taxa de churn  
        📊 Sua previsibilidade de receita  
        📈 O crescimento sustentável do seu negócio  

        **A Ferramenta de Análise Cohort**

        Para facilitar esse processo, disponibilizamos aqui uma ferramenta gratuita em **Streamlit**:

        ✅ Faça sua própria Análise Cohort  
        ✅ Use seus dados reais  
        ✅ Aplique segmentações personalizadas  
        ✅ Tudo no seu ritmo — sem precisar de conhecimento em código  

        **Comece agora e tome decisões mais inteligentes baseadas em dados sólidos!**
        """)

    
    if not df.empty:
        # Métricas rápidas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de Usuários", df[user_id_col].nunique())
        with col2:
            st.metric("Período Analisado", f"{selected_dates[0]} a {selected_dates[1]}")
        
        # Processamento e visualização
        with st.spinner("Gerando análise de cohort..."):
            retention, cohort_pivot = preprocess_data(df, user_id_col, date_col, segment_col)
            
            with col3:
                avg_retention = retention.replace(0, np.nan).mean().mean()
                st.metric("Retenção Média", f"{avg_retention:.1f}%" if not np.isnan(avg_retention) else "N/A")
            
            # Visualização expandida dos dados
            with st.expander("🔍 Visualizar Dados Brutos"):
                st.dataframe(df.sort_values(date_col), height=250, use_container_width=True)

            plot_interactive_cohort(retention, cohort_pivot, display_type, color_palette, segment_col)
    else:
        st.warning("Nenhum dado disponível para análise. Carregue um arquivo ou use os dados de exemplo.")

if __name__ == "__main__":
    main()