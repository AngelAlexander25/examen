import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date, datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración de página optimizada
st.set_page_config(
    page_title="Sistema de Analítica de Reservas de Hotel 🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado mejorado con paleta elegante
st.markdown("""
<style>
    /* Nueva paleta de colores elegante */
    :root {
        --primary-navy: #0F1D3C;
        --secondary-navy: #1A315A;
        --light-blue-gray: #E7EBF2;
        --accent-teal: #4DD599;
        --light-gray: #f0f2f6;
        --dark-gray: #495057;
        --success-green: #28a745;
        --warning-orange: #ffc107;
        --danger-red: #dc3545;
        --purple: #6f42c1;
        --indigo: #6610f2;
    }
    
    /* Estilo de métricas mejoradas */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, var(--light-blue-gray) 0%, #ffffff 100%);
        border: 1px solid #dee2e6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Tarjetas de información */
    .info-card {
        background: linear-gradient(135deg, #ffffff 0%, var(--light-gray) 100%);
        border-left: 4px solid var(--primary-navy);
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    
    .kpi-card {
        background: linear-gradient(135deg, #ffffff 0%, var(--light-blue-gray) 100%);
        border: 2px solid var(--primary-navy);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .executive-card {
        background: linear-gradient(135deg, var(--purple) 0%, var(--indigo) 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    
    .occupancy-card {
        background: linear-gradient(135deg, var(--success-green) 0%, #20c997 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Header mejorado */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, var(--primary-navy) 0%, var(--secondary-navy) 100%);
        color: white;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 20px 20px;
    }
    
    .section-header {
        background: linear-gradient(135deg, var(--secondary-navy) 0%, var(--primary-navy) 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }

    .risk-high {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #f5c6cb;
    }
    .risk-medium {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ffeeba;
    }
    .risk-low {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border: 1px solid #c3e6cb;
    }

    /* Estilo del Expander */
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        border-radius: 8px;
        color: var(--primary-navy);
        border: 1px solid #dee2e6;
        padding: 1rem;
    }
    .streamlit-expanderHeader:hover {
        background-color: #e6e8eb;
    }
    
</style>
""", unsafe_allow_html=True)

# --- Funciones de carga optimizadas ---
@st.cache_resource(show_spinner="Cargando modelo...")
def load_model():
    """Carga solo el modelo de ML"""
    try:
        model = joblib.load('xgboost_model_mejorado.joblib')
        return model
    except FileNotFoundError:
        st.error("❌ Error: 'xgboost_model_mejorado.joblib' no encontrado.")
        st.stop()

@st.cache_resource(show_spinner="Cargando preprocessor...")
def load_preprocessor():
    """Carga solo el preprocessor"""
    try:
        preprocessor = joblib.load('preprocessor_mejorado.joblib')
        return preprocessor
    except FileNotFoundError:
        st.error("❌ Error: 'preprocessor_mejorado.joblib' no encontrado.")
        st.stop()

@st.cache_data(show_spinner="Cargando datos...")
def load_hotel_data(hotel_filter=None):
    """Carga y preprocesa solo los datos necesarios"""
    try:
        df = pd.read_csv("hotel_bookings_clean (1).csv")
        
        # Solo procesar columnas necesarias para visualizaciones
        necessary_cols = ['arrival_date_month', 'country', 'is_canceled', 
                         'adr', 'lead_time', 'reservation_status_date', 'arrival_date',
                         'hotel', 'deposit_type', 'customer_type', 'market_segment',
                         'stays_in_weekend_nights', 'stays_in_week_nights', 'adults',
                         'children', 'babies', 'total_of_special_requests', 'booking_changes']
        
        # Verificar que las columnas existen
        available_cols = [col for col in necessary_cols if col in df.columns]
        df_viz = df[available_cols].copy()
        
        # Filtrar por tipo de hotel si se proporciona un filtro
        if hotel_filter and hotel_filter != 'Todos':
            df_viz = df_viz[df_viz['hotel'] == hotel_filter]
        
        # Optimizar tipos de datos
        if 'is_canceled' in df_viz.columns:
            df_viz['is_canceled'] = df_viz['is_canceled'].astype('int8')
        if 'adr' in df_viz.columns:
            df_viz['adr'] = pd.to_numeric(df_viz['adr'], errors='coerce')
        if 'lead_time' in df_viz.columns:
            df_viz['lead_time'] = pd.to_numeric(df_viz['lead_time'], errors='coerce')
            
        return df_viz
    except FileNotFoundError:
        st.error("❌ Error: 'hotel_bookings_clean (1).csv' no encontrado.")
        st.stop()

# --- Nuevas funciones para análisis detallado ---
@st.cache_data
def calculate_detailed_kpis(df):
    """Calcula KPIs detallados del hotel"""
    kpis = {}
    
    if 'is_canceled' in df.columns and 'adr' in df.columns:
        # KPIs básicos
        kpis['total_bookings'] = len(df)
        kpis['canceled_bookings'] = df['is_canceled'].sum()
        kpis['confirmed_bookings'] = kpis['total_bookings'] - kpis['canceled_bookings']
        kpis['cancellation_rate'] = (kpis['canceled_bookings'] / kpis['total_bookings']) * 100
        
        # KPIs financieros
        kpis['total_potential_revenue'] = df['adr'].sum()
        kpis['lost_revenue'] = df[df['is_canceled'] == 1]['adr'].sum()
        kpis['confirmed_revenue'] = df[df['is_canceled'] == 0]['adr'].sum()
        kpis['revenue_loss_rate'] = (kpis['lost_revenue'] / kpis['total_potential_revenue']) * 100
        
        # KPIs operativos
        kpis['avg_adr'] = df['adr'].mean()
        kpis['avg_adr_canceled'] = df[df['is_canceled'] == 1]['adr'].mean()
        kpis['avg_adr_confirmed'] = df[df['is_canceled'] == 0]['adr'].mean()
        
        if 'lead_time' in df.columns:
            kpis['avg_lead_time'] = df['lead_time'].mean()
            kpis['avg_lead_time_canceled'] = df[df['is_canceled'] == 1]['lead_time'].mean()
        
        # Análisis por segmentos
        if 'market_segment' in df.columns:
            segment_analysis = df.groupby('market_segment').agg({
                'is_canceled': ['count', 'sum', 'mean'],
                'adr': 'mean'
            }).round(2)
            kpis['segment_analysis'] = segment_analysis
    
    return kpis

def predict_occupancy_for_date_range(start_date, end_date):
    """
    Predice ocupación diaria en un rango de fechas.
    Esta función es una simulación para propósitos de demostración.
    """
    
    delta = end_date - start_date
    num_days = delta.days
    
    if num_days <= 0:
        return []

    predictions = []
    current_date = start_date
    
    # Simular una ocupación base que varía estacionalmente
    base_occupancy_factor = 0.8  # 80%
    
    for i in range(num_days):
        # Ocupación base, ajustada con una variación sinusoidal simulando estacionalidad
        day_of_year = (current_date - date(current_date.year, 1, 1)).days
        seasonal_adj = 0.1 * np.sin(2 * np.pi * day_of_year / 365)
        
        # Un factor aleatorio para simular variabilidad
        random_noise = np.random.uniform(-0.05, 0.05)
        
        predicted_occupancy = (base_occupancy_factor + seasonal_adj + random_noise) * 100
        
        # Limitar a valores realistas
        predicted_occupancy = min(98, max(45, predicted_occupancy))
        
        predictions.append({
            'date': current_date,
            'predicted_occupancy': round(predicted_occupancy, 1),
            'confidence_level': max(70, 95 - i * 0.5), # Confianza que decrece con el tiempo
            'recommended_action': get_occupancy_recommendation(predicted_occupancy)
        })
        current_date += timedelta(days=1)
        
    return predictions

def get_occupancy_recommendation(occupancy):
    """Genera recomendaciones basadas en ocupación predicha"""
    if occupancy > 85:
        return "🔴 Considerar estrategia de overbooking moderado"
    elif occupancy > 70:
        return "🟢 Ocupación óptima - Continuar estrategia actual"
    elif occupancy > 50:
        return "🟡 Lanzar promociones de último minuto"
    else:
        return "🔴 Activar campañas de marketing agresivas"

def create_gauge_chart(value):
    """Crea gráfico de medidor optimizado con colores elegantes"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': "Probabilidad de Cancelación (%)", 'font': {'size': 20, 'color': '#0F1D3C'}},
        number={'font': {'size': 28, 'color': '#0F1D3C'}},
        gauge={
            'axis': {'range': [0, 100], 'tickfont': {'size': 14}},
            'bar': {'color': "#1A315A", 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#dee2e6",
            'steps': [
                {'range': [0, 25], 'color': "#d4edda"},
                {'range': [25, 50], 'color': "#fff3cd"},
                {'range': [50, 75], 'color': "#f8d7da"},
                {'range': [75, 100], 'color': "#f5c6cb"}
            ],
            'threshold': {
                'line': {'color': "#dc3545", 'width': 3}, 
                'thickness': 0.8, 
                'value': value
            }
        }))
    fig.update_layout(
        height=350, 
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# --- Funciones de visualización mejoradas ---
def create_executive_dashboard_chart(df):
    """Crea gráfico ejecutivo combinado"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Tendencia de Cancelaciones', 'Ingresos vs Pérdidas', 
                       'Ocupación por Tipo de Hotel', 'Lead Time vs Cancelación'),
        specs=[[{"secondary_y": True}, {"type": "bar"}],
               [{"type": "pie"}, {"type": "scatter"}]]
    )
    
    # Gráfico 1: Tendencia temporal
    if 'arrival_date_month' in df.columns:
        monthly_data = df.groupby('arrival_date_month')['is_canceled'].agg(['count', 'sum']).reset_index()
        fig.add_trace(
            go.Scatter(x=monthly_data['arrival_date_month'], 
                      y=monthly_data['count'], 
                      name="Total Reservas", 
                      line=dict(color='#1A315A')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=monthly_data['arrival_date_month'], 
                      y=monthly_data['sum'], 
                      name="Cancelaciones", 
                      line=dict(color='#dc3545')),
            row=1, col=1, secondary_y=True
        )
    
    # Gráfico 2: Ingresos
    if 'adr' in df.columns and 'is_canceled' in df.columns:
        revenue_data = df.groupby('is_canceled')['adr'].sum().reset_index()
        fig.add_trace(
            go.Bar(x=['Confirmadas', 'Canceladas'], 
                   y=revenue_data['adr'], 
                   marker_color=['#4DD599', '#dc3545'],
                   name="Ingresos"),
            row=1, col=2
        )
    
    # Gráfico 3: Ocupación por tipo de hotel
    if 'hotel' in df.columns:
        hotel_data = df.groupby('hotel')['is_canceled'].agg(['count', 'sum']).reset_index()
        hotel_data['confirmed'] = hotel_data['count'] - hotel_data['sum']
        fig.add_trace(
            go.Pie(labels=hotel_data['hotel'], 
                   values=hotel_data['confirmed'], 
                   name="Ocupación Confirmada"),
            row=2, col=1
        )
    
    # Gráfico 4: Lead time scatter
    if 'lead_time' in df.columns and 'adr' in df.columns:
        sample_data = df.sample(n=min(1000, len(df)))  # Muestra para performance
        colors = ['#dc3545' if x == 1 else '#4DD599' for x in sample_data['is_canceled']]
        fig.add_trace(
            go.Scatter(x=sample_data['lead_time'], 
                      y=sample_data['adr'],
                      mode='markers',
                      marker=dict(color=colors, opacity=0.6),
                      name="Lead Time vs ADR"),
            row=2, col=2
        )
    
    fig.update_layout(height=700, showlegend=True, 
                     title_text="📊 Panel Ejecutivo de Análisis")
    return fig

def create_cancellation_map(df):
    """Crea un mapa mundial de la tasa de cancelación por país."""
    if 'country' not in df.columns or 'is_canceled' not in df.columns:
        return None

    # Limpiar y preparar datos
    country_data = df.groupby('country').agg(
        total_bookings=('is_canceled', 'size'),
        canceled_bookings=('is_canceled', 'sum')
    ).reset_index()
    
    country_data['cancellation_rate'] = (country_data['canceled_bookings'] / country_data['total_bookings']) * 100
    
    # Nueva paleta de colores y ajuste de tamaño
    fig = px.choropleth(
        country_data,
        locations='country',
        color='cancellation_rate',
        hover_name='country',
        hover_data={'cancellation_rate': ':.2f%', 'total_bookings': True, 'canceled_bookings': True},
        color_continuous_scale=px.colors.sequential.PuBu, # Una paleta azul más elegante
        title="🌍 Tasa de Cancelación por País",
        locationmode='ISO-3'
    )
    
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='equirectangular'
        ),
        # Aumentar el tamaño del mapa
        height=600,
    )
    
    return fig

def create_occupancy_prediction_chart(predictions):
    """Crea gráfico de predicción de ocupación"""
    dates = [p['date'] for p in predictions]
    occupancies = [p['predicted_occupancy'] for p in predictions]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates, y=occupancies,
        mode='lines+markers',
        name='Ocupación Predicha',
        line=dict(color='#4DD599', width=4), # Color turquesa elegante
        marker=dict(size=10, color='#4DD599')
    ))
    
    fig.update_layout(
        title="🏨 Predicción de Ocupación Diaria",
        xaxis_title="Fecha",
        yaxis_title="Ocupación (%)",
        height=400,
        yaxis=dict(range=[0, 100]),
        xaxis_tickformat='%b %d'
    )
    
    return fig

def create_occupancy_breakdown_chart(predictions):
    """Crea gráfico de barras para la distribución de ocupación"""
    occupancy_levels = {'Baja (45-60%)': 0, 'Media (61-80%)': 0, 'Alta (81-100%)': 0}
    colors = {'Baja (45-60%)': '#dc3545', 'Media (61-80%)': '#ffc107', 'Alta (81-100%)': '#4DD599'}

    for p in predictions:
        if 45 <= p['predicted_occupancy'] <= 60:
            occupancy_levels['Baja (45-60%)'] += 1
        elif 61 <= p['predicted_occupancy'] <= 80:
            occupancy_levels['Media (61-80%)'] += 1
        else:
            occupancy_levels['Alta (81-100%)'] += 1
            
    level_names = list(occupancy_levels.keys())
    day_counts = list(occupancy_levels.values())
    bar_colors = [colors[name] for name in level_names]

    fig = go.Figure(go.Bar(
        x=level_names,
        y=day_counts,
        marker_color=bar_colors,
        text=day_counts,
        textposition='outside'
    ))

    fig.update_layout(
        title="📊 Distribución de Ocupación por Nivel",
        xaxis_title="Nivel de Ocupación",
        yaxis_title="Número de Días",
        yaxis=dict(showgrid=False),
        xaxis=dict(showgrid=False)
    )
    
    return fig

def create_detailed_kpi_charts(kpis):
    """Crea gráficos detallados de KPIs"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Distribución de Reservas', 'Análisis Financiero', 
                       'Comparación ADR', 'Análisis por Segmento'),
        specs=[[{'type': 'pie'}, {'type': 'bar'}], 
               [{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # Gráfico 1: Distribución de reservas
    fig.add_trace(go.Pie(
        labels=['Confirmadas', 'Canceladas'],
        values=[kpis['confirmed_bookings'], kpis['canceled_bookings']],
        marker_colors=['#4DD599', '#dc3545'], # Colores actualizados
        name="Reservas"
    ), row=1, col=1)
    
    # Gráfico 2: Análisis financiero
    fig.add_trace(go.Bar(
        x=['Ingresos Confirmados', 'Ingresos Perdidos'],
        y=[kpis['confirmed_revenue'], kpis['lost_revenue']],
        marker_color=['#4DD599', '#dc3545'], # Colores actualizados
        name="Ingresos"
    ), row=1, col=2)
    
    # Gráfico 3: Comparación ADR
    fig.add_trace(go.Bar(
        x=['ADR General', 'ADR Canceladas', 'ADR Confirmadas'],
        y=[kpis['avg_adr'], kpis['avg_adr_canceled'], kpis['avg_adr_confirmed']],
        marker_color=['#0F1D3C', '#dc3545', '#4DD599'], # Colores actualizados
        name="ADR Promedio"
    ), row=2, col=1)
    
    # Gráfico 4: Análisis por segmento (nuevo)
    if 'segment_analysis' in kpis and not kpis['segment_analysis'].empty:
        segment_df = kpis['segment_analysis'].copy()
        segment_df.columns = ['Total_Reservas', 'Cancelaciones', 'Tasa_Cancelacion', 'ADR_Promedio']
        
        fig.add_trace(go.Bar(
            x=segment_df.index,
            y=segment_df['Tasa_Cancelacion'] * 100,
            marker_color='#1A315A', # Color actualizado
            name="Tasa de Cancelación por Segmento"
        ), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False,
                     title_text="📈 Análisis Detallado de KPIs")
    return fig

# --- Header principal ---
st.markdown("""
<div class="main-header">
    <h1>🏨 Sistema Avanzado de Analítica Hotelera</h1>
    <p>Plataforma integral de análisis predictivo y gestión de ocupación</p>
</div>
""", unsafe_allow_html=True)

# Filtro de Hotel en la barra lateral
st.sidebar.markdown("### ⚙️ Filtrar por Hotel")
selected_hotel = st.sidebar.selectbox("Selecciona el Tipo de Hotel", ['Todos', 'City Hotel', 'Resort Hotel'])


# Inicializar session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# Crear tabs principales
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dashboard Ejecutivo", 
    "📈 KPIs Detallados", 
    "🔮 Predicción Cancelaciones",
    "🏨 Predictor de Ocupación"
])

with tab1:
    st.markdown("""
    <div class="section-header">
        <h2>🎯 Panel de Análisis Ejecutivo</h2>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("¿Qué es esta sección?"):
        st.write("""
        El **Dashboard Ejecutivo** ofrece una visión panorámica y estratégica del rendimiento del hotel. Aquí puedes ver métricas clave como el total de reservas, la tasa de cancelación y los ingresos, junto con gráficos que visualizan las principales tendencias. Esta sección está diseñada para ayudarte a tomar decisiones rápidas e informadas a nivel gerencial.
        """)
        
    try:
        df_data = load_hotel_data(selected_hotel)
        kpis = calculate_detailed_kpis(df_data)
        
        # Métricas ejecutivas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Reservas", f"{kpis['total_bookings']:,}")
        with col2:
            st.metric("❌ Cancelaciones", f"{kpis['canceled_bookings']:,}",
                     delta=f"{kpis['cancellation_rate']:.1f}%")
        with col3:
            st.metric("💰 Ingresos Confirmados", f"${kpis['confirmed_revenue']:,.0f}")
        with col4:
            st.metric("📉 Pérdidas por Cancelación", f"${kpis['lost_revenue']:,.0f}",
                     delta=f"-{kpis['revenue_loss_rate']:.1f}%")
        
        # Panel ejecutivo combinado
        st.subheader("📈 Análisis Multidimensional")
        exec_chart = create_executive_dashboard_chart(df_data)
        st.plotly_chart(exec_chart, use_container_width=True)
        
        # Mapa de cancelaciones
        st.subheader("🌍 Mapa de Tasa de Cancelación por País")
        cancellation_map = create_cancellation_map(df_data)
        if cancellation_map:
            st.plotly_chart(cancellation_map, use_container_width=True)
        
        # Análisis por segmentos
        st.divider()
        st.subheader("🔍 Resumen Ejecutivo")
        col_res1, col_res2 = st.columns(2)
        if 'segment_analysis' in kpis and not kpis['segment_analysis'].empty:
            segment_df = kpis['segment_analysis'].copy()
            segment_df.columns = ['Total_Reservas', 'Cancelaciones', 'Tasa_Cancelacion', 'ADR_Promedio']
            
            worst_segment = segment_df['Tasa_Cancelacion'].idxmax()
            best_segment = segment_df['Tasa_Cancelacion'].idxmin()

            with col_res1:
                st.markdown(f"""
                <div class="risk-high">
                    <h4>Segmento de Mayor Riesgo</h4>
                    <p><strong>{worst_segment}</strong></p>
                    <p>Con una tasa de cancelación de <strong>{segment_df.loc[worst_segment, 'Tasa_Cancelacion']:.1f}%</strong>, este segmento requiere una atención inmediata para mitigar las pérdidas.</p>
                </div>
                """, unsafe_allow_html=True)
            with col_res2:
                st.markdown(f"""
                <div class="risk-low">
                    <h4>Segmento Más Sólido</h4>
                    <p><strong>{best_segment}</strong></p>
                    <p>Con una baja tasa de cancelación de <strong>{segment_df.loc[best_segment, 'Tasa_Cancelacion']:.1f}%</strong>, este segmento demuestra una fuerte lealtad y estabilidad en las reservas.</p>
                </div>
                """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error cargando dashboard ejecutivo: {e}")

with tab2:
    st.markdown("""
    <div class="section-header">
        <h2>📊 Análisis Detallado de KPIs</h2>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("¿Qué son los KPIs detallados?"):
        st.write("""
        Los **KPIs (Indicadores Clave de Rendimiento)** detallados ofrecen una inmersión profunda en las métricas más importantes del hotel. Aquí puedes analizar el impacto financiero de las cancelaciones, comparar las tarifas promedio y segmentar el rendimiento por tipo de cliente o mercado.
        """)

    try:
        df_data = load_hotel_data(selected_hotel)
        kpis = calculate_detailed_kpis(df_data)
        
        # KPIs financieros detallados
        st.subheader("💰 Análisis Financiero Detallado")
        
        col_fin1, col_fin2, col_fin3 = st.columns(3)
        with col_fin1:
            st.markdown("""
            <div class="kpi-card">
                <h4>📈 Ingresos Totales</h4>
                <h2>$%s</h2>
                <p>Potencial de ingresos totales</p>
            </div>
            """ % f"{kpis['total_potential_revenue']:,.0f}", unsafe_allow_html=True)
        
        with col_fin2:
            st.markdown("""
            <div class="kpi-card">
                <h4>✅ Ingresos Confirmados</h4>
                <h2>$%s</h2>
                <p>Ingresos garantizados</p>
            </div>
            """ % f"{kpis['confirmed_revenue']:,.0f}", unsafe_allow_html=True)
        
        with col_fin3:
            st.markdown("""
            <div class="kpi-card">
                <h4>❌ Pérdidas por Cancelación</h4>
                <h2>$%s</h2>
                <p>%.1f%% del total</p>
            </div>
            """ % (f"{kpis['lost_revenue']:,.0f}", kpis['revenue_loss_rate']), unsafe_allow_html=True)
        
        # Gráficos detallados de KPIs
        detailed_kpi_chart = create_detailed_kpi_charts(kpis)
        st.plotly_chart(detailed_kpi_chart, use_container_width=True)
        
        # Análisis por segmentos
        if 'segment_analysis' in kpis and not kpis['segment_analysis'].empty:
            st.subheader("🎯 Análisis por Segmento de Mercado")
            
            segment_df = kpis['segment_analysis'].copy()
            segment_df.columns = ['Total_Reservas', 'Cancelaciones', 'Tasa_Cancelacion', 'ADR_Promedio']
            segment_df['Tasa_Cancelacion'] = segment_df['Tasa_Cancelacion'] * 100
            
            st.dataframe(segment_df.round(2), use_container_width=True)
            
            # Recomendaciones por segmento
            st.markdown("**🎯 Recomendaciones por Segmento:**")
            worst_segment = segment_df['Tasa_Cancelacion'].idxmax()
            best_segment = segment_df['Tasa_Cancelacion'].idxmin()
            
            col_seg1, col_seg2 = st.columns(2)
            with col_seg1:
                st.error(f"⚠️ **Atención:** {worst_segment} tiene la mayor tasa de cancelación ({segment_df.loc[worst_segment, 'Tasa_Cancelacion']:.1f}%)")
            with col_seg2:
                st.success(f"✅ **Fortaleza:** {best_segment} tiene la menor tasa de cancelación ({segment_df.loc[best_segment, 'Tasa_Cancelacion']:.1f}%)")
        
        # KPIs operativos adicionales
        st.subheader("⚙️ Métricas Operativas")
        
        col_op1, col_op2, col_op3 = st.columns(3)
        with col_op1:
            st.metric("🏷️ ADR Promedio General", f"${kpis['avg_adr']:.2f}")
        with col_op2:
            st.metric("📅 Lead Time Promedio", f"{kpis.get('avg_lead_time', 0):.0f} días")
        with col_op3:
            efficiency = (kpis['confirmed_bookings'] / kpis['total_bookings']) * 100
            st.metric("⚡ Eficiencia de Conversión", f"{efficiency:.1f}%")
        
    except Exception as e:
        st.error(f"Error cargando KPIs detallados: {e}")

with tab3:
    st.markdown("""
    <div class="section-header">
        <h2>🔮 Predictor de Cancelaciones</h2>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("¿Cómo funciona esta herramienta?"):
        st.write("""
        Esta herramienta te permite predecir la probabilidad de que una reserva específica sea cancelada, utilizando un modelo de Machine Learning. Al ingresar los detalles de una reserva (como la antelación, el tipo de depósito y el segmento de mercado), el modelo analiza el riesgo y te proporciona una puntuación de probabilidad. Esto te ayuda a identificar reservas de alto riesgo y a tomar medidas proactivas para retener a los huéspedes.
        """)
    
    # Usar formulario para evitar recargas
    with st.form("prediction_form"):
        st.subheader("📝 Ingresa los datos de la reserva")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🏨 Información Básica**")
            # Este selectbox ya existía, pero ahora el filtro global hace el trabajo
            hotel = st.selectbox("Tipo de Hotel", ["Resort Hotel", "City Hotel"])
            lead_time = st.number_input("Antelación (días)", min_value=0, value=30)
            stays_in_week_nights = st.number_input("Noches entre semana", min_value=0, value=2)
            stays_in_weekend_nights = st.number_input("Noches fin de semana", min_value=0, value=1)
            adults = st.number_input("Adultos", min_value=1, value=2)
            children = st.number_input("Niños", min_value=0, value=0)
            babies = st.number_input("Bebés", min_value=0, value=0)
        
        with col2:
            st.markdown("**💼 Detalles de Reserva**")
            adr = st.number_input("Tarifa Diaria (ADR)", min_value=0.0, value=100.0)
            total_of_special_requests = st.number_input("Peticiones especiales", min_value=0, value=0)
            booking_changes = st.number_input("Cambios en reserva", min_value=0, value=0)
            days_in_waiting_list = st.number_input("Días en lista de espera", min_value=0, value=0)
            previous_cancellations = st.number_input("Cancelaciones anteriores", min_value=0, value=0)
            previous_bookings_not_canceled = st.number_input("Reservas anteriores exitosas", min_value=0, value=0)
            is_repeated_guest = st.checkbox("Huésped repetido")
        
        with col3:
            st.markdown("**📊 Información Comercial**")
            market_segment = st.selectbox("Segmento", ["Online TA", "Offline TA/TO", "Groups", "Direct", "Corporate"])
            distribution_channel = st.selectbox("Canal", ["TA/TO", "Direct", "Corporate", "GDS"])
            customer_type = st.selectbox("Tipo Cliente", ["Transient", "Contract", "Transient-Party", "Group"])
            deposit_type = st.selectbox("Tipo Depósito", ["No Deposit", "Non Refund", "Refundable"])
            reserved_room_type = st.selectbox("Habitación Reservada", ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
            assigned_room_type = st.selectbox("Habitación Asignada", ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
        
        # Botón de predicción con estilo personalizado
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔮 PREDECIR PROBABILIDAD DE CANCELACIÓN", use_container_width=True)
        
        if submitted:
            try:
                # Calcular características derivadas
                total_guests = adults + children + babies
                length_of_stay = stays_in_week_nights + stays_in_weekend_nights
                cancellation_history_rate = previous_cancellations / (previous_cancellations + previous_bookings_not_canceled + 1)
                is_room_changed = 1 if reserved_room_type != assigned_room_type else 0
                is_weekend_stay = 1 if stays_in_weekend_nights > 0 else 0
                
                # Preparar datos de entrada
                feature_cols = ['lead_time', 'length_of_stay', 'total_guests', 'cancellation_history_rate', 'adr', 
                               'days_in_waiting_list', 'hotel', 'deposit_type', 'customer_type', 'market_segment', 
                               'distribution_channel', 'reserved_room_type', 'assigned_room_type', 
                               'is_repeated_guest', 'is_room_changed', 'is_weekend_stay', 'total_of_special_requests']
                
                input_data = {
                    'lead_time': lead_time,
                    'length_of_stay': length_of_stay,
                    'total_guests': total_guests,
                    'cancellation_history_rate': cancellation_history_rate,
                    'adr': adr,
                    'days_in_waiting_list': days_in_waiting_list,
                    'hotel': hotel,
                    'deposit_type': deposit_type,
                    'customer_type': customer_type,
                    'market_segment': market_segment,
                    'distribution_channel': distribution_channel,
                    'reserved_room_type': reserved_room_type,
                    'assigned_room_type': assigned_room_type,
                    'is_repeated_guest': is_repeated_guest,
                    'is_room_changed': is_room_changed,
                    'is_weekend_stay': is_weekend_stay,
                    'total_of_special_requests': total_of_special_requests
                }
                
                input_df = pd.DataFrame([input_data])[feature_cols]
                
                # Cargar modelo y preprocessor
                model = load_model()
                preprocessor = load_preprocessor()
                
                # Realizar predicción
                with st.spinner('🔄 Analizando datos y calculando probabilidad...'):
                    processed_data = preprocessor.transform(input_df)
                    prediction_proba = model.predict_proba(processed_data)[:, 1][0] * 100
                
                # Guardar resultado en session state
                st.session_state.prediction_result = prediction_proba
                st.session_state.prediction_made = True
                # Guardar datos relevantes para el análisis de factores
                st.session_state.input_summary = {
                    'lead_time': lead_time,
                    'deposit_type': deposit_type,
                    'is_repeated_guest': is_repeated_guest,
                    'is_room_changed': is_room_changed,
                    'days_in_waiting_list': days_in_waiting_list,
                    'cancellation_history_rate': cancellation_history_rate,
                    'length_of_stay': length_of_stay,
                    'total_of_special_requests': total_of_special_requests
                }
                
            except Exception as e:
                st.error(f"❌ Error en la predicción: {e}")
                st.session_state.prediction_made = False
    
    # Mostrar resultado si existe
    if st.session_state.prediction_made and st.session_state.prediction_result is not None:
        st.divider()
        
        # Header de resultados con estilo
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #e7ebf2 0%, #ffffff 100%); 
                    border-radius: 15px; margin: 2rem 0; border: 2px solid #0F1D3C;">
            <h2 style="color: #0F1D3C; margin: 0; font-size: 1.8rem;">✨ Resultado de la Predicción</h2>
            <p style="color: #495057; margin: 0.5rem 0 0 0;">Análisis completo de riesgo de cancelación</p>
        </div>
        """, unsafe_allow_html=True)
        
        prediction_value = st.session_state.prediction_result
        
        # Layout principal de resultados
        col_gauge, col_details = st.columns([1, 1])
        
        with col_gauge:
            # Gráfico de medidor principal
            gauge_fig = create_gauge_chart(prediction_value)
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        with col_details:
            # Métricas de resultado
            st.subheader("📊 Métricas de Riesgo")
            
            # Métrica principal
            st.metric(
                label="🎯 Probabilidad de Cancelación", 
                value=f"{prediction_value:.1f}%",
                delta=f"{prediction_value - 37.2:.1f}% vs promedio histórico"
            )
            
            # Clasificación de riesgo con estilos
            if prediction_value > 75:
                st.markdown("""
                <div class="risk-high">
                    <h3>🚨 RIESGO MUY ALTO</h3>
                    <p><strong>Acción inmediata requerida:</strong><br>
                    • Contactar al huésped en las próximas 24h<br>
                    • Ofrecer incentivos (descuentos, upgrades)<br>
                    • Confirmar disponibilidad de fechas alternativas</p>
                </div>
                """, unsafe_allow_html=True)
            elif prediction_value > 50:
                st.markdown("""
                <div class="risk-medium">
                    <h3>⚠️ RIESGO ALTO</h3>
                    <p><strong>Estrategias de retención:</strong><br>
                    • Email personalizado de confirmación<br>
                    • Información sobre servicios del hotel<br>
                    • Flexibilidad en políticas de cambio</p>
                </div>
                """, unsafe_allow_html=True)
            elif prediction_value > 25:
                st.markdown("""
                <div class="risk-medium">
                    <h3>ℹ️ RIESGO MODERADO</h3>
                    <p><strong>Monitoreo recomendado:</strong><br>
                    • Seguimiento periódico<br>
                    • Email de cortesía una semana antes<br>
                    • Preparar plan de contingencia</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="risk-low">
                    <h3>✅ RIESGO BAJO</h3>
                    <p><strong>Reserva estable:</strong><br>
                    • Continuar con proceso estándar<br>
                    • Email de bienvenida 48h antes<br>
                    • Preparar experiencia de llegada</p>
                </div>
                """, unsafe_allow_html=True)

with tab4:
    st.markdown("""
    <div class="section-header">
        <h2>🏨 Predictor de Ocupación</h2>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("¿Qué es este predictor?"):
        st.write("""
        Este predictor te ayuda a visualizar la ocupación estimada para un rango de fechas. Basado en tendencias históricas de ocupación y cancelaciones, la herramienta simula la ocupación diaria, permitiéndote planificar con anticipación y ajustar estrategias de marketing o precios para maximizar la rentabilidad del hotel.
        """)
        
    st.subheader("Selecciona el rango de fechas para la predicción")
    
    col_date1, col_date2 = st.columns(2)
    
    with col_date1:
        start_date = st.date_input("Fecha de inicio", value=date.today())
    
    with col_date2:
        end_date = st.date_input("Fecha de finalización", value=date.today() + timedelta(days=30))
        
    if start_date >= end_date:
        st.error("La fecha de finalización debe ser posterior a la fecha de inicio.")
    else:
        try:
            st.divider()
            
            # Generar predicciones
            predictions = predict_occupancy_for_date_range(start_date, end_date)
            
            # Gráfico de predicción
            occupancy_chart = create_occupancy_prediction_chart(predictions)
            st.plotly_chart(occupancy_chart, use_container_width=True)
            
            # Tarjetas de resumen
            avg_occupancy = sum(p['predicted_occupancy'] for p in predictions) / len(predictions)
            
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("Ocupación Promedio", f"{avg_occupancy:.1f}%")
            with col_info2:
                peak_occupancy = max(p['predicted_occupancy'] for p in predictions)
                st.metric("Ocupación Máxima", f"{peak_occupancy:.1f}%")
            with col_info3:
                low_occupancy = min(p['predicted_occupancy'] for p in predictions)
                st.metric("Ocupación Mínima", f"{low_occupancy:.1f}%")

            st.divider()

            # Gráfico de desglose de ocupación
            occupancy_breakdown_chart = create_occupancy_breakdown_chart(predictions)
            st.plotly_chart(occupancy_breakdown_chart, use_container_width=True)

            st.divider()
            
            # Tabla detallada de predicciones
            with st.expander("📋 Ver Análisis Detallado por Día"):
                for pred in predictions:
                    col_pred1, col_pred2 = st.columns(2)
                    
                    with col_pred1:
                        st.markdown(f"""
                        <div class="occupancy-card">
                            <h4>📈 Métricas Predichas</h4>
                            <p><strong>Ocupación:</strong> {pred['predicted_occupancy']}%</p>
                            <p><strong>Confianza:</strong> {pred['confidence_level']}%</p>
                            <p><strong>Estado:</strong> {'🟢 Óptimo' if pred['predicted_occupancy'] > 70 else '🟡 Moderado' if pred['predicted_occupancy'] > 50 else '🔴 Crítico'}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_pred2:
                        st.markdown("**🎯 Recomendación:**")
                        st.info(pred['recommended_action'])
                        
                        # Acciones específicas basadas en ocupación
                        if pred['predicted_occupancy'] > 85:
                            st.markdown("""
                            **Acciones sugeridas:**
                            • Implementar overbooking controlado (5-10%)
                            • Aumentar tarifas dinámicamente
                            • Preparar lista de espera
                            • Coordinar con hoteles aliados
                            """)
                        elif pred['predicted_occupancy'] > 70:
                            st.markdown("""
                            **Acciones sugeridas:**
                            • Mantener tarifas actuales
                            • Monitorear reservas de último minuto
                            • Optimizar personal de servicio
                            • Preparar upselling
                            """)
                        elif pred['predicted_occupancy'] > 50:
                            st.markdown("""
                            **Acciones sugeridas:**
                            • Lanzar promociones de último minuto
                            • Activar campañas en redes sociales
                            • Contactar agencias de viaje
                            • Ofrecer paquetes especiales
                            """)
                        else:
                            st.markdown("""
                            **Acciones urgentes:**
                            • Campañas agresivas de marketing
                            • Descuentos significativos (20-30%)
                            • Contactar grupos corporativos
                            • Considerar cierre parcial de áreas
                            """)

        except Exception as e:
            st.error(f"Error en predictor de ocupación: {e}")

# Footer mejorado
st.divider()
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #6c757d; font-size: 0.9rem;">
    <p>🏨 Sistema Avanzado de Analítica Hotelera | Powered by Machine Learning & Predictive Analytics</p>
    <p>Maximizando rentabilidad a través de predicciones inteligentes de cancelaciones y ocupación</p>
</div>
""", unsafe_allow_html=True)
