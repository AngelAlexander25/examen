import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date
import plotly.express as px
import plotly.graph_objects as go

# Configuración de página optimizada
st.set_page_config(
    page_title="Sistema de Analítica de Reservas de Hotel 🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar el estilo
st.markdown("""
<style>
    /* Colores principales */
    :root {
        --primary-blue: #1f4e79;
        --secondary-blue: #2e5984;
        --light-blue: #e8f4fd;
        --accent-teal: #0f4c75;
        --light-gray: #f8f9fa;
        --dark-gray: #495057;
        --success-green: #28a745;
        --warning-orange: #ffc107;
        --danger-red: #dc3545;
    }
    
    /* Estilo de métricas mejoradas */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, var(--light-blue) 0%, #ffffff 100%);
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
        border-left: 4px solid var(--primary-blue);
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    
    .info-card h3 {
        color: var(--primary-blue);
        margin-top: 0;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    .info-card p {
        color: var(--dark-gray);
        line-height: 1.6;
        margin-bottom: 0;
    }
    
    /* Estilo de pestañas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: linear-gradient(135deg, var(--light-blue) 0%, #ffffff 100%);
        border-radius: 8px;
        border: 1px solid #dee2e6;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--secondary-blue) 100%);
        color: white;
    }
    
    /* Formulario mejorado */
    .stForm {
        background: var(--light-gray);
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #dee2e6;
    }
    
    /* Botones personalizados */
    .prediction-button {
        background: linear-gradient(135deg, var(--accent-teal) 0%, var(--primary-blue) 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.8rem 2rem;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .prediction-button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(15, 76, 117, 0.3);
    }
    
    /* Alertas personalizadas */
    .risk-high {
        background: linear-gradient(135deg, #ffe6e6 0%, #ffcccc 100%);
        border-left: 4px solid var(--danger-red);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border-left: 4px solid var(--warning-orange);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 4px solid var(--success-green);
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Header mejorado */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--secondary-blue) 100%);
        color: white;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 20px 20px;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
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
def load_hotel_data():
    """Carga y preprocesa solo los datos necesarios"""
    try:
        df = pd.read_csv("hotel_bookings_clean (1).csv")
        
        # Solo procesar columnas necesarias para visualizaciones
        necessary_cols = ['arrival_date_month', 'country', 'is_canceled', 
                         'adr', 'lead_time', 'reservation_status_date', 'arrival_date']
        
        # Verificar que las columnas existen
        available_cols = [col for col in necessary_cols if col in df.columns]
        df_viz = df[available_cols].copy()
        
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

@st.cache_data
def get_feature_importance():
    """Calcula importancia de características una sola vez"""
    model = load_model()
    preprocessor = load_preprocessor()
    
    try:
        # Obtener nombres de características del preprocessor
        cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(
            input_features=['hotel', 'deposit_type', 'customer_type', 'market_segment', 
                           'distribution_channel', 'reserved_room_type', 'assigned_room_type'])
        num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
        bool_features = preprocessor.named_transformers_['bool'].get_feature_names_out()
        
        all_features = list(num_features) + list(cat_features) + list(bool_features)
        
        feature_importances = pd.DataFrame({
            'feature': all_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        return feature_importances
    except Exception as e:
        st.warning(f"No se pudo calcular la importancia: {e}")
        return pd.DataFrame({'feature': [], 'importance': []})

# --- Funciones de visualización mejoradas ---
def create_gauge_chart(value):
    """Crea gráfico de medidor optimizado con colores elegantes"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': "Probabilidad de Cancelación (%)", 'font': {'size': 20, 'color': '#1f4e79'}},
        number={'font': {'size': 28, 'color': '#1f4e79'}},
        gauge={
            'axis': {'range': [0, 100], 'tickfont': {'size': 14}},
            'bar': {'color': "#1f4e79", 'thickness': 0.3},
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

@st.cache_data
def create_monthly_chart(df):
    """Crea gráfico mensual con estilo mejorado"""
    if 'arrival_date_month' not in df.columns or 'is_canceled' not in df.columns:
        return None
    
    ordered_months = ['January', 'February', 'March', 'April', 'May', 'June', 
                     'July', 'August', 'September', 'October', 'November', 'December']
    
    df_copy = df.copy()
    df_copy['arrival_date_month'] = pd.Categorical(df_copy['arrival_date_month'], 
                                                  categories=ordered_months, ordered=True)
    
    monthly_cancel_rate = df_copy.groupby('arrival_date_month', observed=True)['is_canceled'].mean().reset_index()
    monthly_cancel_rate['is_canceled'] = monthly_cancel_rate['is_canceled'] * 100
    
    fig = px.area(monthly_cancel_rate, x='arrival_date_month', y='is_canceled', 
                  title="📅 Tasa de Cancelación por Mes",
                  labels={'arrival_date_month': 'Mes', 'is_canceled': 'Tasa de Cancelación (%)'},
                  color_discrete_sequence=['#1f4e79'])
    fig.update_traces(fill='tonexty', fillcolor='rgba(31, 78, 121, 0.3)')
    fig.update_layout(
        height=400,
        title_font_size=16,
        title_font_color='#1f4e79',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
    )
    return fig

@st.cache_data
def create_country_chart(df):
    """Crea gráfico de países con gradiente elegante"""
    if 'country' not in df.columns or 'is_canceled' not in df.columns:
        return None
    
    country_cancel_count = df[df['is_canceled'] == 1]['country'].value_counts().nlargest(10).reset_index()
    
    fig = px.bar(country_cancel_count, x='count', y='country', orientation='h',
                 title="🌍 Top 10 Países con Más Cancelaciones",
                 labels={'count': 'Número de Cancelaciones', 'country': 'País'},
                 color='count',
                 color_continuous_scale=['#e8f4fd', '#1f4e79'])
    fig.update_yaxes(categoryorder='total ascending')
    fig.update_layout(
        height=400,
        title_font_size=16,
        title_font_color='#1f4e79',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
    )
    return fig

@st.cache_data
def create_adr_distribution_chart(df):
    """Crea gráfico de distribución de ADR"""
    if 'adr' not in df.columns:
        return None
    
    fig = px.histogram(df, x='adr', nbins=50,
                       title="💰 Distribución de Tarifa Diaria Promedio (ADR)",
                       labels={'adr': 'ADR ($)', 'count': 'Frecuencia'},
                       color_discrete_sequence=['#2e5984'])
    fig.update_layout(
        height=350,
        title_font_size=16,
        title_font_color='#1f4e79',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
    )
    return fig

@st.cache_data
def create_lead_time_chart(df):
    """Crea gráfico de tiempo de anticipación"""
    if 'lead_time' not in df.columns or 'is_canceled' not in df.columns:
        return None
    
    # Crear rangos de lead time
    df_copy = df.copy()
    df_copy['lead_time_range'] = pd.cut(df_copy['lead_time'], 
                                       bins=[0, 30, 90, 180, 365, 999],
                                       labels=['0-30 días', '31-90 días', '91-180 días', 
                                              '181-365 días', '+365 días'])
    
    lead_time_cancel = df_copy.groupby('lead_time_range')['is_canceled'].mean().reset_index()
    lead_time_cancel['is_canceled'] = lead_time_cancel['is_canceled'] * 100
    
    fig = px.bar(lead_time_cancel, x='lead_time_range', y='is_canceled',
                 title="⏰ Tasa de Cancelación por Tiempo de Anticipación",
                 labels={'lead_time_range': 'Rango de Anticipación', 'is_canceled': 'Tasa de Cancelación (%)'},
                 color='is_canceled',
                 color_continuous_scale=['#e8f4fd', '#1f4e79'])
    fig.update_layout(
        height=350,
        title_font_size=16,
        title_font_color='#1f4e79',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
    )
    return fig

# --- Funciones para tarjetas informativas ---
def show_model_info_card():
    """Muestra tarjeta informativa sobre el modelo"""
    with st.expander("🤖 ¿Cómo funciona nuestro modelo de predicción?", expanded=False):
        st.markdown("""
        <div class="info-card">
            <h3>Modelo de Machine Learning XGBoost</h3>
            <p><strong>Algoritmo:</strong> Utilizamos XGBoost (Extreme Gradient Boosting), uno de los algoritmos más potentes para problemas de clasificación.</p>
            
            <h3>Variables Principales</h3>
            <ul>
                <li><strong>Tiempo de Anticipación:</strong> Días entre la reserva y la llegada</li>
                <li><strong>Tipo de Depósito:</strong> Si el huésped pagó depósito o no</li>
                <li><strong>Historial del Cliente:</strong> Cancelaciones y reservas anteriores</li>
                <li><strong>Características de la Reserva:</strong> Duración, huéspedes, habitación</li>
                <li><strong>Canal de Distribución:</strong> Cómo llegó el cliente</li>
            </ul>
            
            <h3>Precisión del Modelo</h3>
            <p>Nuestro modelo alcanza una precisión del <strong>87%</strong> en la predicción de cancelaciones, permitiendo tomar decisiones proactivas para mejorar la rentabilidad del hotel.</p>
        </div>
        """, unsafe_allow_html=True)

def show_kpi_info_card():
    """Muestra tarjeta informativa sobre los KPIs"""
    with st.expander("📊 Entendiendo los KPIs del Dashboard", expanded=False):
        st.markdown("""
        <div class="info-card">
            <h3>Indicadores Clave de Rendimiento (KPIs)</h3>
            
            <p><strong>Total de Reservas:</strong> Número total de reservas registradas en el sistema.</p>
            
            <p><strong>Cancelaciones:</strong> Cantidad de reservas que fueron canceladas por los huéspedes.</p>
            
            <p><strong>Tasa de Cancelación:</strong> Porcentaje de reservas canceladas sobre el total. Una tasa normal oscila entre 25-40%.</p>
            
            <h3>¿Por qué son importantes?</h3>
            <ul>
                <li><strong>Planificación de Ingresos:</strong> Permite ajustar estrategias de pricing y overbooking</li>
                <li><strong>Gestión de Recursos:</strong> Optimiza personal y servicios según demanda real</li>
                <li><strong>Estrategias de Retención:</strong> Identifica patrones para reducir cancelaciones</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def show_interpretation_guide():
    """Muestra guía de interpretación de resultados"""
    with st.expander("🎯 Guía de Interpretación de Resultados", expanded=False):
        st.markdown("""
        <div class="info-card">
            <h3>Niveles de Riesgo de Cancelación</h3>
            
            <div class="risk-low">
                <strong>🟢 RIESGO BAJO (0-25%):</strong> Reserva muy estable. Continúa con el proceso normal.
            </div>
            
            <div class="risk-medium">
                <strong>🟡 RIESGO MODERADO (26-50%):</strong> Monitorear la reserva. Considerar email de confirmación.
            </div>
            
            <div class="risk-medium">
                <strong>🟠 RIESGO ALTO (51-75%):</strong> Implementar estrategias de retención inmediatas.
            </div>
            
            <div class="risk-high">
                <strong>🔴 RIESGO MUY ALTO (+75%):</strong> Contacto urgente con el huésped. Ofrecer incentivos.
            </div>
            
            <h3>Acciones Recomendadas</h3>
            <ul>
                <li><strong>Contacto Proactivo:</strong> Llamada o email personalizado</li>
                <li><strong>Incentivos:</strong> Descuentos, upgrades de habitación, servicios adicionales</li>
                <li><strong>Flexibilidad:</strong> Opciones de cambio de fecha sin penalización</li>
                <li><strong>Confirmación:</strong> Solicitar confirmación de asistencia</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# --- Header principal ---
st.markdown("""
<div class="main-header">
    <h1>🏨 Sistema de Analítica de Reservas</h1>
    <p>Herramienta inteligente de análisis y predicción de cancelaciones hoteleras</p>
</div>
""", unsafe_allow_html=True)

# Inicializar session state para evitar recargas
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# Crear tabs con estilo mejorado
tab1, tab2 = st.tabs(["📈 Dashboard de Análisis", "🔮 Predicción de Cancelaciones"])

with tab1:
    st.header("📊 Análisis de Métricas y Tendencias")
    
    # Tarjeta informativa sobre KPIs
    show_kpi_info_card()
    
    # Cargar datos solo cuando se necesiten
    try:
        df_data = load_hotel_data()
        
        # Mostrar métricas básicas con estilo mejorado (solo los 3 originales)
        st.subheader("📋 Indicadores Principales")
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        
        with col_metric1:
            total_bookings = len(df_data)
            st.metric("📊 Total Reservas", f"{total_bookings:,}")
        
        with col_metric2:
            if 'is_canceled' in df_data.columns:
                canceled = df_data['is_canceled'].sum()
                st.metric("❌ Cancelaciones", f"{canceled:,}")
            
        with col_metric3:
            if 'is_canceled' in df_data.columns:
                cancel_rate = (df_data['is_canceled'].mean() * 100)
                st.metric("📈 Tasa Cancelación", f"{cancel_rate:.1f}%")
        
        st.divider()
        
        # Gráficas principales
        st.subheader("📈 Análisis de Tendencias")
        col1, col2 = st.columns(2)
        
        with col1:
            monthly_fig = create_monthly_chart(df_data)
            if monthly_fig:
                st.plotly_chart(monthly_fig, use_container_width=True)
            else:
                st.info("Datos de meses no disponibles")
        
        with col2:
            country_fig = create_country_chart(df_data)
            if country_fig:
                st.plotly_chart(country_fig, use_container_width=True)
            else:
                st.info("Datos de países no disponibles")
        
        # Segunda fila de gráficos
        col3, col4 = st.columns(2)
        
        with col3:
            adr_fig = create_adr_distribution_chart(df_data)
            if adr_fig:
                st.plotly_chart(adr_fig, use_container_width=True)
            else:
                st.info("Datos de ADR no disponibles")
        
        with col4:
            lead_time_fig = create_lead_time_chart(df_data)
            if lead_time_fig:
                st.plotly_chart(lead_time_fig, use_container_width=True)
            else:
                st.info("Datos de tiempo de anticipación no disponibles")
        
        # Gráfico de importancia de características
        st.subheader("🎯 Variables más Importantes del Modelo")
        feature_imp = get_feature_importance()
        if not feature_imp.empty:
            fig_imp = px.bar(feature_imp, x='importance', y='feature', orientation='h',
                            title="🔍 Importancia de Variables en la Predicción",
                            color='importance',
                            color_continuous_scale=['#e8f4fd', '#1f4e79'])
            fig_imp.update_yaxes(categoryorder='total ascending')
            fig_imp.update_layout(
                height=400,
                title_font_size=16,
                title_font_color='#1f4e79',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)')
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        else:
            st.info("No se pudo cargar la importancia de características")
            
    except Exception as e:
        st.error(f"Error cargando datos para visualización: {e}")

with tab2:
    st.header("🔮 Predictor de Cancelaciones")
    
    # Tarjetas informativas
    show_model_info_card()
    show_interpretation_guide()
    
    # Usar formulario para evitar recargas
    with st.form("prediction_form"):
        st.subheader("📝 Ingresa los datos de la reserva")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🏨 Información Básica**")
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
                               'is_repeated_guest', 'is_room_changed', 'is_weekend_stay']
                
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
                    'is_weekend_stay': is_weekend_stay
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
        <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #e8f4fd 0%, #ffffff 100%); 
                    border-radius: 15px; margin: 2rem 0; border: 2px solid #1f4e79;">
            <h2 style="color: #1f4e79; margin: 0; font-size: 1.8rem;">✨ Resultado de la Predicción</h2>
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
            
            # Clasificación de riesgo con estilos - DEBUG
            st.write(f"**DEBUG:** Valor de predicción = {prediction_value:.2f}%")
            
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
        
        # Sección de factores de riesgo
        st.subheader("🔍 Análisis de Factores de Riesgo")
        
        col_factors1, col_factors2 = st.columns(2)
        
        with col_factors1:
            st.markdown("**📈 Factores que Aumentan el Riesgo:**")
            risk_factors = []
            
            if st.session_state.input_summary['lead_time'] > 120:
                risk_factors.append("• Reserva con mucha antelación (+120 días)")
            if st.session_state.input_summary['deposit_type'] == 'No Deposit':
                risk_factors.append("• Sin depósito pagado")
            if st.session_state.input_summary['is_room_changed']:
                risk_factors.append("• Cambio de tipo de habitación")
            if st.session_state.input_summary['days_in_waiting_list'] > 0:
                risk_factors.append("• Estuvo en lista de espera")
            if st.session_state.input_summary['cancellation_history_rate'] > 0.3:
                risk_factors.append("• Historial de cancelaciones alto")
            
            if risk_factors:
                for factor in risk_factors[:5]:  # Máximo 5 factores
                    st.write(factor)
            else:
                st.success("✅ Pocos factores de riesgo identificados")
        
        with col_factors2:
            st.markdown("**📉 Factores que Reducen el Riesgo:**")
            protection_factors = []
            
            if st.session_state.input_summary['is_repeated_guest']:
                protection_factors.append("• Cliente repetido")
            if st.session_state.input_summary['deposit_type'] != 'No Deposit':
                protection_factors.append("• Depósito pagado")
            if st.session_state.input_summary['lead_time'] < 30:
                protection_factors.append("• Reserva de último momento")
            if st.session_state.input_summary['total_of_special_requests'] > 0:
                protection_factors.append("• Peticiones especiales realizadas")
            if st.session_state.input_summary['length_of_stay'] > 3:
                protection_factors.append("• Estancia prolongada")
            
            if protection_factors:
                for factor in protection_factors[:5]:  # Máximo 5 factores
                    st.write(factor)
            else:
                st.info("ℹ️ Pocos factores de protección identificados")
        
        # Sección de recomendaciones
        st.subheader("💡 Recomendaciones Personalizadas")
        
        recommendations_col1, recommendations_col2 = st.columns(2)
        
        with recommendations_col1:
            st.markdown("**🎯 Acciones Inmediatas:**")
            if prediction_value > 75:
                st.write("1. 📞 Llamada telefónica dentro de 24h")
                st.write("2. 💰 Ofrecer descuento del 10-15%")
                st.write("3. 🏨 Upgrade de habitación gratuito")
                st.write("4. 📅 Flexibilidad total en fechas")
            elif prediction_value > 50:
                st.write("1. ✉️ Email personalizado inmediato")
                st.write("2. 🎁 Servicios adicionales gratuitos")
                st.write("3. 📋 Confirmar detalles de la reserva")
                st.write("4. 💬 Seguimiento en 48-72h")
            else:
                st.write("1. ✉️ Email de bienvenida estándar")
                st.write("2. 📱 SMS recordatorio 48h antes")
                st.write("3. 🏨 Información sobre servicios")
                st.write("4. 🎉 Preparar experiencia de llegada")
        
        with recommendations_col2:
            st.markdown("**📊 Seguimiento Sugerido:**")
            if prediction_value > 50:
                st.write("• Monitoreo diario hasta la llegada")
                st.write("• Registro de todas las interacciones")
                st.write("• Escalación a manager si no responde")
                st.write("• Preparar plan de overbooking")
            else:
                st.write("• Seguimiento semanal rutinario")
                st.write("• Email automático 1 semana antes")
                st.write("• Proceso de check-in estándar")
                st.write("• Solicitar feedback post-estancia")
        
        # Botones de acción
        st.divider()
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if st.button("🔄 Nueva Predicción", use_container_width=True):
                st.session_state.prediction_made = False
                st.session_state.prediction_result = None
                st.rerun()
        
        with col_btn2:
            if st.button("📧 Generar Email", use_container_width=True):
                st.info("Funcionalidad de generación de emails próximamente")
        
        with col_btn3:
            if st.button("📊 Exportar Reporte", use_container_width=True):
                st.info("Funcionalidad de exportación próximamente")

# Footer elegante
st.divider()
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #6c757d; font-size: 0.9rem;">
    <p>🏨 Sistema de Analítica de Reservas | Powered by Machine Learning</p>
    <p>Mejorando la rentabilidad hotelera a través de predicciones inteligentes</p>
</div>
""", unsafe_allow_html=True)
