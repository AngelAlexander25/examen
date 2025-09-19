import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date, timedelta
import plotly.express as px
import plotly.graph_objects as go

# Streamlit page configuration must be the first command
st.set_page_config(
    page_title="Sistema de Analítica de Reservas de Hotel 🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Asset Loading Functions ---
@st.cache_resource
def load_assets():
    """Loads the pre-trained model and preprocessor."""
    try:
        model = joblib.load('xgboost_model_mejorado.joblib')
        preprocessor = joblib.load('preprocessor_mejorado.joblib')
        return model, preprocessor
    except FileNotFoundError:
        st.error("❌ Error: 'xgboost_model_mejorado.joblib' or 'preprocessor_mejorado.joblib' files not found. Please ensure they are in the same directory.")
        st.stop()

@st.cache_data
def load_and_prepare_data():
    """Loads and preprocesses the hotel data for visualization."""
    try:
        df = pd.read_csv("hotel_bookings_clean (1).csv")
        # Ensure 'arrival_date_month' is an ordered categorical for plotting
        ordered_months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        df['arrival_date_month'] = pd.Categorical(df['arrival_date_month'], categories=ordered_months, ordered=True)

        # Prepare data for new graphs
        df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
        df['arrival_date'] = pd.to_datetime(df['arrival_date'])
        
        # Calculate days to cancellation
        df['days_to_cancellation'] = (df['reservation_status_date'] - df['arrival_date']).dt.days
        df.loc[df['is_canceled'] == 0, 'days_to_cancellation'] = np.nan
        
        return df
    except FileNotFoundError:
        st.error("❌ Error: 'hotel_bookings_clean (1).csv' file not found. Please ensure it is in the same directory.")
        st.stop()

# --- Functions for visualizations ---
def display_gauge_chart(value):
    """Displays a gauge chart for the prediction probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': "Probabilidad de Cancelación", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "black"},
            'bar': {'color': "white"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 25], 'color': "green"},
                {'range': [25, 50], 'color': "yellow"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': value}}))
    fig.update_layout(height=250, width=500, margin=dict(l=10, r=10, t=5, b=5))
    return fig

def get_feature_importance(model, preprocessor):
    """
    Retrieves feature importance from the model and maps it to original feature names.
    """
    # Get preprocessor feature names
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(input_features=['hotel', 'deposit_type', 'customer_type', 'market_segment', 'distribution_channel', 'reserved_room_type', 'assigned_room_type'])
    num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
    bool_features = preprocessor.named_transformers_['bool'].get_feature_names_out()
    
    # Combine feature names
    all_features = list(num_features) + list(cat_features) + list(bool_features)
    
    # Create a DataFrame for importance
    feature_importances = pd.DataFrame({
        'feature': all_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    return feature_importances

# Load assets
model, preprocessor = load_assets()
# Load data for the charts
df_data = load_and_prepare_data()
feature_importances = get_feature_importance(model, preprocessor)

# --- Main Title and Description ---
st.title("Sistema de Analítica de Reservas de Hotel 🏨📊")
st.markdown(
    """
    **Bienvenido.** Esta herramienta te ayuda a **analizar** las tendencias de reservas y a **predecir**
    la probabilidad de que una reserva sea cancelada.
    """
)
st.sidebar.header("Menú de Navegación")

# --- Application Tabs ---
tab1, tab2 = st.tabs(["Dashboard de Análisis 📈", "Formulario de Predicción 🔮"])

with tab1:
    st.header("Análisis de Métricas y Tendencias 📉")
    with st.expander("Información del Dashboard"):
        st.write("💡 **Dashboard de Análisis:** En esta sección, puedes ver los principales indicadores de éxito (KPIs) del proyecto. Las gráficas te ayudan a entender los patrones históricos y los factores que más influyen en las cancelaciones, como el tipo de hotel o la antelación de la reserva.")

    st.markdown("---")
    
    # Visualization Section
    st.subheader("Gráficas de Tendencias de Cancelación")

    col_vis_1, col_vis_2 = st.columns(2)
    with col_vis_1:
        # Tasa de Cancelación por Mes
        st.markdown("**Tasa de Cancelación por Mes**")
        with st.expander("Ver descripción"):
            st.write("Esta gráfica de líneas te permite identificar si hay patrones estacionales o meses del año en los que las cancelaciones son más frecuentes. Es útil para planificar estrategias de retención en periodos de alto riesgo.")
        # FIX: Added observed=False to fix the FutureWarning
        monthly_cancel_rate = df_data.groupby('arrival_date_month', observed=False)['is_canceled'].mean().reset_index()
        monthly_cancel_rate['is_canceled'] = monthly_cancel_rate['is_canceled'] * 100
        fig_monthly = px.line(monthly_cancel_rate, x='arrival_date_month', y='is_canceled', markers=True,
                              labels={'arrival_date_month': 'Mes', 'is_canceled': 'Tasa de Cancelación (%)'})
        st.plotly_chart(fig_monthly, use_container_width=True)

    with col_vis_2:
        # Top 10 Países con Más Cancelaciones
        st.markdown("**Top 10 Países con Más Cancelaciones**")
        with st.expander("Ver descripción"):
            st.write("El gráfico de barras muestra los países con la mayor cantidad de reservas canceladas. Esta información es útil para ajustar campañas de marketing y ofertas personalizadas según la procedencia del huésped.")
        country_cancel_count = df_data[df_data['is_canceled'] == 1]['country'].value_counts().nlargest(10).reset_index()
        fig_country = px.bar(country_cancel_count, x='count', y='country', orientation='h',
                             labels={'count': 'Número de Cancelaciones', 'country': 'País'})
        fig_country.update_yaxes(categoryorder='total ascending')
        st.plotly_chart(fig_country, use_container_width=True)

    st.markdown("---")

    # --- New Section for Advanced Analysis ---
    st.subheader("Análisis Avanzado de Factores de Riesgo")

    col_vis_3, col_vis_4 = st.columns(2)
    with col_vis_3:
        # Importancia de las Características
        st.markdown("**Importancia de las Variables del Modelo**")
        with st.expander("Ver descripción"):
            st.write("Esta gráfica muestra cuáles son las variables que más influyen en la predicción del modelo. Te ayuda a entender `por qué` se toma una decisión y a enfocar tus análisis en los factores más críticos.")
        fig_importance = px.bar(feature_importances, x='importance', y='feature', orientation='h',
                                labels={'importance': 'Importancia', 'feature': 'Variable'})
        fig_importance.update_yaxes(categoryorder='total ascending')
        st.plotly_chart(fig_importance, use_container_width=True)
    
    with col_vis_4:
        # Análisis de ADR (Ingresos)
        st.markdown("**Relación entre Ingresos (ADR) y Cancelación**")
        with st.expander("Ver descripción"):
            st.write("Este gráfico de violín muestra cómo la tarifa diaria promedio (ADR) se distribuye para las reservas canceladas y no canceladas. Es útil para entender el impacto financiero de las cancelaciones.")
        fig_adr = px.violin(df_data, y="adr", x="is_canceled", color="is_canceled", box=True, points="all",
                            labels={'is_canceled': 'Cancelado', 'adr': 'Tarifa Diaria Promedio'})
        st.plotly_chart(fig_adr, use_container_width=True)
    
    st.markdown("---")

    col_vis_5, col_vis_6 = st.columns(2)
    with col_vis_5:
        # Análisis del tiempo hasta la cancelación
        st.markdown("**Distribución del Tiempo hasta la Cancelación**")
        with st.expander("Ver descripción"):
            st.write("Esta gráfica muestra el número de días que pasaron entre la fecha de la reserva y la fecha de cancelación. Es crucial para identificar una 'ventana de riesgo' y saber cuándo es más importante contactar al huésped.")
        fig_time_to_cancel = px.histogram(df_data[df_data['is_canceled']==1], x='days_to_cancellation', nbins=50,
                                         labels={'days_to_cancellation': 'Días hasta la Cancelación', 'count': 'Número de Cancelaciones'})
        st.plotly_chart(fig_time_to_cancel, use_container_width=True)

    with col_vis_6:
        # Antelación de la Reserva vs. Cancelación (reubicada)
        st.markdown("**Antelación de la Reserva vs. Cancelación**")
        with st.expander("Ver descripción"):
            st.write("Un `lead time` largo a menudo se correlaciona con un mayor riesgo de cancelación, ya que los planes pueden cambiar. Esta gráfica lo muestra claramente y es una de las variables más importantes del modelo.")
        fig_violin = px.violin(df_data, y="lead_time", x="is_canceled", color="is_canceled", box=True, points="all",
                               hover_data=df_data.columns,
                               labels={'is_canceled': 'Cancelado', 'lead_time': 'Antelación (días)'})
        st.plotly_chart(fig_violin, use_container_width=True)

with tab2:
    st.header("Formulario de Predicción de Cancelaciones 🔮")
    with st.expander("Información sobre el Formulario"):
        st.write("💡 **Formulario de Predicción:** Esta sección te permite ingresar los datos de una reserva individual. El sistema procesará esta información y usará el modelo de Machine Learning para calcular la probabilidad de cancelación.")
        st.write("En la variable `is_canceled`, el valor **1** significa **'Cancelado'** y el valor **0** significa **'No Cancelado'** (el huésped se presentó). La predicción que verás será un valor entre 0 y 100%, que representa la probabilidad de que el valor sea 1.")

    with st.expander("Ingresar Detalles de la Reserva", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Información Básica")
            hotel = st.selectbox("Tipo de Hotel", ["Resort Hotel", "City Hotel"])
            lead_time = st.number_input("Antelación (días)", min_value=0, value=30)
            stays_in_week_nights = st.number_input("Noches de Lunes a Viernes", min_value=0, value=2)
            stays_in_weekend_nights = st.number_input("Noches de Fin de Semana", min_value=0, value=1)
            total_of_special_requests = st.number_input("Peticiones Especiales", min_value=0, value=0)
            adults = st.number_input("Número de Adultos", min_value=0, value=1)
            children = st.number_input("Número de Niños", min_value=0, value=0)
            babies = st.number_input("Número de Bebés", min_value=0, value=0)
            is_repeated_guest = st.selectbox("Huésped Repetido", [False, True])


        with col2:
            st.subheader("Información de Huéspedes y Tarifas")
            adr = st.number_input("Tarifa Diaria Promedio (ADR)", min_value=0.0, value=100.0)
            booking_changes = st.number_input("Cambios en la Reserva", min_value=0, value=0)
            days_in_waiting_list = st.number_input("Días en Lista de Espera", min_value=0, value=0)
            previous_cancellations = st.number_input("Cancelaciones Anteriores", min_value=0, value=0)
            previous_bookings_not_canceled = st.number_input("Reservas Anteriores No Canceladas", min_value=0, value=0)
            reserved_room_type = st.selectbox("Habitación Reservada", ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'P'])
            assigned_room_type = st.selectbox("Habitación Asignada", ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'P'])
            customer_type = st.selectbox("Tipo de Cliente", ["Transient", "Contract", "Transient-Party", "Group"])


        with col3:
            st.subheader("Detalles de la Reserva")
            country = st.selectbox("País", ["PRT", "GBR", "ESP", "IRL", "FRA", "USA", "DEU", "ITA", "BRA", "NLD", "Unknown"])
            market_segment = st.selectbox("Segmento de Mercado", ["Online TA", "Offline TA/TO", "Groups", "Direct", "Corporate", "Complementary", "Aviation", "Undefined"])
            distribution_channel = st.selectbox("Canal de Distribución", ["TA/TO", "Direct", "Corporate", "Undefined", "GDS"])
            deposit_type = st.selectbox("Tipo de Depósito", ["No Deposit", "Non Refund", "Refundable"])
            arrival_date_year = st.number_input("Año de Llegada", min_value=2015, value=date.today().year)
            arrival_date_month = st.selectbox("Mes de Llegada", ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
            arrival_date_day_of_month = st.number_input("Día de Llegada", min_value=1, value=date.today().day)

    if st.button("PREDECIR PROBABILIDAD DE CANCELACIÓN", help="Haz clic para obtener la predicción", use_container_width=True):
        st.spinner("Calculando la predicción...")

        # Feature Engineering: Crear las mismas variables que en el script de entrenamiento
        total_guests = adults + children + babies
        length_of_stay = stays_in_week_nights + stays_in_weekend_nights
        cancellation_history_rate = previous_cancellations / (previous_cancellations + previous_bookings_not_canceled + 1)
        is_room_changed = 1 if reserved_room_type != assigned_room_type else 0
        is_weekend_stay = 1 if stays_in_weekend_nights > 0 else 0

        # Crear el DataFrame de entrada con las mismas columnas que el modelo espera
        # El orden y los nombres de las columnas son CRUCIALES.
        input_data = {
            'hotel': [hotel],
            'lead_time': [lead_time],
            'stays_in_week_nights': [stays_in_week_nights],
            'stays_in_weekend_nights': [stays_in_weekend_nights],
            'adults': [adults],
            'children': [children],
            'babies': [babies],
            'is_repeated_guest': [is_repeated_guest],
            'previous_cancellations': [previous_cancellations],
            'previous_bookings_not_canceled': [previous_bookings_not_canceled],
            'booking_changes': [booking_changes],
            'days_in_waiting_list': [days_in_waiting_list],
            'adr': [adr],
            'total_of_special_requests': [total_of_special_requests],
            'length_of_stay': [length_of_stay],
            'is_weekend_stay': [is_weekend_stay],
            'total_guests': [total_guests],
            'cancellation_history_rate': [cancellation_history_rate],
            'is_room_changed': [is_room_changed],
            'market_segment': [market_segment],
            'distribution_channel': [distribution_channel],
            'customer_type': [customer_type],
            'deposit_type': [deposit_type],
            'reserved_room_type': [reserved_room_type],
            'assigned_room_type': [assigned_room_type]
        }
        
        # Crear el DataFrame con las columnas que el modelo espera
        feature_cols = ['lead_time', 'length_of_stay', 'total_guests', 'cancellation_history_rate', 'adr', 
                        'days_in_waiting_list', 'hotel', 'deposit_type', 'customer_type', 'market_segment', 
                        'distribution_channel', 'reserved_room_type', 'assigned_room_type', 
                        'is_repeated_guest', 'is_room_changed', 'is_weekend_stay']
        
        # Creamos el DataFrame y nos aseguramos de que el orden de las columnas sea el mismo que el del entrenamiento
        input_df = pd.DataFrame(input_data)
        input_df = input_df[feature_cols]

        try:
            processed_data = preprocessor.transform(input_df)
            prediction_proba = model.predict_proba(processed_data)[:, 1][0] * 100
            
            st.markdown("---")
            st.subheader("✨ Resultado de la Predicción")
            
            # Display the gauge chart
            gauge_fig = display_gauge_chart(prediction_proba)
            st.plotly_chart(gauge_fig, use_container_width=True)

            if prediction_proba > 50:
                st.error("🚨 Riesgo Alto: Esta reserva podría ser cancelada. Se recomienda una gestión proactiva.")
            else:
                st.success("✅ Riesgo Bajo: La probabilidad de cancelación es baja.")
        
        except Exception as e:
            st.error(f"❌ Ocurrió un error al procesar la predicción: {e}")
            st.info("Por favor, revisa que los valores ingresados sean correctos.")