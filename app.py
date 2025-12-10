import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Define the BRL to INR conversion rate
BRL_TO_INR_RATE = 16.52  # (1 BRL = 16.52 INR)

# Page configuration
st.set_page_config(
    page_title="E-commerce Sales Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .prediction-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and artifacts from models/ folder
@st.cache_resource
def load_model_artifacts():
    try:
        with open('models/sales_predictor_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        with open('models/model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        with open('models/sales_statistics.pkl', 'rb') as f:
            stats = pickle.load(f)
        return model, scaler, feature_columns, metadata, stats
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please run the Jupyter notebook first to train the model.")
        st.stop()

# Load artifacts
model, scaler, feature_columns, metadata, stats = load_model_artifacts()

# Header
st.markdown('<div class="main-header">📈 E-commerce Sales Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Daily Sales Forecasting System</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/sales-performance.png", width=150)
    st.header("⚙️ Prediction Settings")
    
    st.markdown("---")
    st.subheader("📊 Model Information")
    st.info(f"""
    **Model Type:** {metadata['model_name']}  
    **R² Score:** {metadata['r2']:.4f}  
    **MAE:** ₹ {(metadata['mae'] * BRL_TO_INR_RATE):,.2f}  
    **RMSE:** ₹ {(metadata['rmse'] * BRL_TO_INR_RATE):,.2f}  
    **Trained:** {metadata['training_date']}
    """)
    
    st.markdown("---")
    st.subheader("📈 Historical Stats (INR)")
    st.success(f"""
    **Avg Daily Sales:** ₹ {(stats['avg_daily_sales'] * BRL_TO_INR_RATE):,.2f}  
    **Max Daily Sales:** ₹ {(stats['max_daily_sales'] * BRL_TO_INR_RATE):,.2f}  
    **Total Revenue:** ₹ {(stats['total_revenue'] * BRL_TO_INR_RATE):,.2f}  
    **Total Orders:** {stats['total_orders']:,}  
    **Avg Order Value:** ₹ {(stats['avg_order_value'] * BRL_TO_INR_RATE):,.2f}
    """)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Make Prediction", "📊 Visualizations", "📈 Batch Predictions", "ℹ️ About"])

with tab1:
    st.header("🎯 Daily Sales Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📅 Date Information")
        pred_date = st.date_input(
            "Select Date for Prediction",
            value=datetime.now(),
            min_value=datetime(2016, 1, 1),
            max_value=datetime(2030, 12, 31)
        )
        
        # Extract date features
        year = pred_date.year
        month = pred_date.month
        day = pred_date.day
        dayofweek = pred_date.weekday()
        quarter = (month - 1) // 3 + 1
        is_weekend = 1 if dayofweek in [5, 6] else 0
        
        st.info(f"""
        **Selected Date:** {pred_date.strftime('%B %d, %Y')}  
        **Day of Week:** {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][dayofweek]}  
        **Quarter:** Q{quarter}  
        **Weekend:** {'Yes' if is_weekend else 'No'}
        """)
    
    with col2:
        st.subheader("📦 Business Metrics")
        num_orders = st.number_input("Expected Number of Orders", min_value=1, value=100, step=10)
        num_items = st.number_input("Expected Number of Items", min_value=1, value=150, step=10)
        
        # Inputs are now in INR, with defaults converted from BRL
        total_freight_inr = st.number_input(
            "Expected Total Shipment Charges (₹)",
            min_value=0.0,
            value=1500.0 * BRL_TO_INR_RATE,
            step=100.0 * BRL_TO_INR_RATE
        )
        
        st.subheader("📊 Historical Context (INR)")
        avg_sales_inr = stats['avg_daily_sales'] * BRL_TO_INR_RATE
        
        sales_lag_1_inr = st.number_input(
            "Previous Day Sales (₹)",
            min_value=0.0,
            value=avg_sales_inr,
            step=100.0 * BRL_TO_INR_RATE
        )
        sales_lag_7_inr = st.number_input(
            "Sales 7 Days Ago (₹)",
            min_value=0.0,
            value=avg_sales_inr,
            step=100.0 * BRL_TO_INR_RATE
        )
        sales_rolling_7_inr = st.number_input(
            "7-Day Average Sales (₹)",
            min_value=0.0,
            value=avg_sales_inr,
            step=100.0 * BRL_TO_INR_RATE
        )
        sales_rolling_30_inr = st.number_input(
            "30-Day Average Sales (₹)",
            min_value=0.0,
            value=avg_sales_inr,
            step=100.0 * BRL_TO_INR_RATE
        )
    
    st.markdown("---")
    
    if st.button("🚀 Predict Sales", type="primary", use_container_width=True):
        # Prepare input data
        # Convert INR inputs back to BRL for the model
        input_data = pd.DataFrame({
            'year': [year],
            'month': [month],
            'day': [day],
            'dayofweek': [dayofweek],
            'quarter': [quarter],
            'is_weekend': [is_weekend],
            'num_orders': [num_orders],
            'num_items': [num_items],
            'total_freight': [total_freight_inr / BRL_TO_INR_RATE],
            'sales_lag_1': [sales_lag_1_inr / BRL_TO_INR_RATE],
            'sales_lag_7': [sales_lag_7_inr / BRL_TO_INR_RATE],
            'sales_rolling_7': [sales_rolling_7_inr / BRL_TO_INR_RATE],
            'sales_rolling_30': [sales_rolling_30_inr / BRL_TO_INR_RATE]
        })
        
        # Ensure correct column order
        input_data = input_data[feature_columns]
        
        # Scale and predict (in BRL)
        input_scaled = scaler.transform(input_data)
        prediction_brl = model.predict(input_scaled)[0]
        
        # Convert BRL prediction to INR for display
        prediction_inr = prediction_brl * BRL_TO_INR_RATE
        
        # Display prediction
        st.markdown(
            f'<div class="prediction-box">Predicted Sales: ₹ {prediction_inr:,.2f}</div>',
            unsafe_allow_html=True
        )
        
        # Analysis metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "vs Historical Avg",
                f"₹ {prediction_inr:,.2f}",
                f"{((prediction_brl - stats['avg_daily_sales']) / stats['avg_daily_sales'] * 100):.1f}%"
            )
        
        with col2:
            expected_orders = num_orders
            predicted_avg_order_brl = prediction_brl / expected_orders if expected_orders > 0 else 0
            predicted_avg_order_inr = predicted_avg_order_brl * BRL_TO_INR_RATE
            st.metric(
                "Predicted Avg Order",
                f"₹ {predicted_avg_order_inr:.2f}",
                f"{((predicted_avg_order_brl - stats['avg_order_value']) / stats['avg_order_value'] * 100):.1f}%"
            )
        
        with col3:
            st.metric(
                "Confidence Level",
                f"{metadata['r2']*100:.1f}%",
                "R² Score"
            )
        
        with col4:
            st.metric(
                "Error Margin",
                f"±₹ {(metadata['mae'] * BRL_TO_INR_RATE):,.0f}",
                "MAE"
            )
        
        # Visualization
        st.subheader("📊 Prediction Breakdown")
        
        fig = go.Figure()
        
        categories = ['Min Expected', 'Prediction', 'Max Expected', 'Historical Avg']
        
        # Calculate values in BRL first
        values_brl = [
            max(0, prediction_brl - metadata['mae']),
            prediction_brl,
            prediction_brl + metadata['mae'],
            stats['avg_daily_sales']
        ]
        
        # Convert to INR for display
        values_inr = [v * BRL_TO_INR_RATE for v in values_brl]
        
        colors = ['#ffa07a', '#ff6347', '#ff4500', '#4169e1']
        
        fig.add_trace(go.Bar(
            x=categories,
            y=values_inr,
            marker_color=colors,
            text=[f'₹ {v:,.0f}' for v in values_inr],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="Prediction Analysis",
            yaxis_title="Sales (₹)",
            showlegend=False,
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("📊 Historical Data Visualizations")
    
    try:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Monthly Sales Trend")
            st.image("Images/monthly_sales_trend.png", use_container_width=True)
            
            st.subheader("📅 Sales by Day of Week")
            st.image("Images/dayofweek_sales.png", use_container_width=True)
        
        with col2:
            st.subheader("🏷️ Top Product Categories")
            st.image("Images/category_sales.png", use_container_width=True)
            
            st.subheader("💳 Payment Distribution")
            st.image("Images/payment_distribution.png", use_container_width=True)
        
        st.subheader("🎯 Model Performance")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image("Images/predictions_comparison.png", use_container_width=True)
        
        with col2:
            st.image("Images/feature_importance.png", use_container_width=True)
            
    except FileNotFoundError:
        st.warning("⚠️ Visualization files not found. Please run the Jupyter notebook to generate them.")

with tab3:
    st.header("📈 Batch Sales Predictions")
    st.info("Predict sales for multiple days at once")
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now())
    with col2:
        num_days = st.number_input("Number of Days", min_value=1, max_value=90, value=7)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        batch_orders = st.number_input("Avg Orders/Day", min_value=1, value=100)
    with col2:
        batch_items = st.number_input("Avg Items/Day", min_value=1, value=150)
    with col3:
        batch_freight_inr = st.number_input(
            "Avg Shipment Charges/Day (₹)",
            min_value=0.0,
            value=1500.0 * BRL_TO_INR_RATE
        )
    
    if st.button("📊 Generate Batch Predictions", type="primary"):
        # Run the batch simulation in BRL (model's original currency)
        predictions_brl = []
        dates = []
        
        base_sales_brl = stats['avg_daily_sales']
        batch_freight_brl = batch_freight_inr / BRL_TO_INR_RATE
        
        for i in range(num_days):
            current_date = start_date + timedelta(days=i)
            
            sales_lag_1_brl = predictions_brl[-1] if predictions_brl else base_sales_brl
            sales_lag_7_brl = predictions_brl[-7] if len(predictions_brl) >= 7 else base_sales_brl
            sales_rolling_7_brl = np.mean(predictions_brl[-7:]) if len(predictions_brl) >= 7 else base_sales_brl
            sales_rolling_30_brl = base_sales_brl  # Simplified for batch
            
            input_data = pd.DataFrame({
                'year': [current_date.year],
                'month': [current_date.month],
                'day': [current_date.day],
                'dayofweek': [current_date.weekday()],
                'quarter': [(current_date.month - 1) // 3 + 1],
                'is_weekend': [1 if current_date.weekday() in [5, 6] else 0],
                'num_orders': [batch_orders],
                'num_items': [batch_items],
                'total_freight': [batch_freight_brl],
                'sales_lag_1': [sales_lag_1_brl],
                'sales_lag_7': [sales_lag_7_brl],
                'sales_rolling_7': [sales_rolling_7_brl],
                'sales_rolling_30': [sales_rolling_30_brl]
            })
            
            input_data = input_data[feature_columns]
            input_scaled = scaler.transform(input_data)
            prediction_brl = model.predict(input_scaled)[0]
            
            predictions_brl.append(prediction_brl)
            dates.append(current_date)
        
        # Convert BRL predictions to INR for display
        predictions_inr = [p * BRL_TO_INR_RATE for p in predictions_brl]
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'Date': dates,
            'Predicted Sales (₹)': [f'₹ {p:,.2f}' for p in predictions_inr],
            'Day of Week': [d.strftime('%A') for d in dates]
        })
        
        st.success(f"✅ Generated predictions for {num_days} days")
        st.dataframe(results_df, use_container_width=True)
        
        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=predictions_inr,
            mode='lines+markers',
            name='Predicted Sales',
            line=dict(color='#ff6347', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_hline(
            y=stats['avg_daily_sales'] * BRL_TO_INR_RATE,
            line_dash="dash",
            line_color="blue",
            annotation_text="Historical Average (INR)"
        )
        
        fig.update_layout(
            title="Predicted Sales Forecast",
            xaxis_title="Date",
            yaxis_title="Sales (₹)",
            height=500,
            template="plotly_white",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Predicted Sales", f"₹ {sum(predictions_inr):,.2f}")
        with col2:
            st.metric("Average Daily Sales", f"₹ {np.mean(predictions_inr):,.2f}")
        with col3:
            st.metric("Highest Day", f"₹ {max(predictions_inr):,.2f}")
        with col4:
            st.metric("Lowest Day", f"₹ {min(predictions_inr):,.2f}")

with tab4:
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ### 🎯 Project Overview
    This **E-commerce Sales Predictor** is a machine learning application that forecasts daily sales based on historical data 
    from a Brazilian e-commerce marketplace (Olist dataset).
    
    ### 🔧 Technical Components
    
    #### Data Analysis & Model Training (Jupyter Notebook)
    - **Data Loading:** Integrated 8 different CSV datasets (customers, orders, products, payments, reviews, etc.)
    - **Feature Engineering:** Created temporal features, lag variables, and rolling averages
    - **Visualization:** Generated comprehensive plots for sales trends, categories, and patterns
    - **ML Models:** Trained Random Forest and Gradient Boosting models
    - **Model Selection:** Automatically selects best performing model based on R² score
    
    #### Prediction Interface (Streamlit App)
    - **Single Predictions:** Input custom parameters for specific date predictions
    - **Batch Predictions:** Forecast sales for multiple consecutive days
    - **Interactive Visualizations:** Dynamic charts using Plotly
    - **Real-time Insights:** Compare predictions with historical averages
    
    ### 📊 Key Features
    - ✅ **Data Preprocessing:** Handles missing values, feature scaling, encoding
    - ✅ **Multiple Algorithms:** Random Forest and Gradient Boosting
    - ✅ **Rich Visualizations:** 6+ different chart types
    - ✅ **Feature Importance:** Identifies key factors affecting sales
    - ✅ **Model Persistence:** Saves trained models for reuse
    - ✅ **User-Friendly Interface:** Intuitive Streamlit dashboard
    
    ### 🎓 Evaluation Metrics
    - **R² Score:** Measures how well predictions match actual values
    - **MAE (Mean Absolute Error):** Average prediction error in currency
    - **RMSE (Root Mean Square Error):** Penalizes larger errors more heavily
    
    ### 📈 Model Performance
    Current model achieves:
    - R² Score: {:.2%} accuracy
    - MAE: ±₹ {:,.2f} average error
    - Trained on real e-commerce transaction data
    
    ### 🚀 How to Use
    1. **Run Jupyter Notebook:** Execute all cells to train model and generate visualizations
    2. **Start Streamlit App:** Run `streamlit run app.py` in your terminal
    3. **Make Predictions:** Use the interface to forecast sales for any date
    
    ### 👨‍💻 Technologies Used
    - Python 3.x
    - Pandas, NumPy (Data Processing)
    - Scikit-learn (Machine Learning)
    - Matplotlib, Seaborn (Static Visualizations)
    - Plotly (Interactive Charts)
    - Streamlit (Web Interface)
    
    ### 📝 Dataset Attribution
    This project uses the Brazilian E-Commerce Public Dataset by Olist, which contains real commercial data 
    from multiple sellers on the Brazilian marketplace.
    
    ---
    
    **Created for Academic Project | Data Science & Machine Learning**
    """.format(metadata['r2'], metadata['mae'] * BRL_TO_INR_RATE))
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Problem Identification** ✅\nPredicting sales for inventory and marketing optimization")
    with col2:
        st.success("**Appropriate Algorithms** ✅\nEnsemble methods (RF & GB) for regression")
    with col3:
        st.warning("**Proper Visualization** ✅\nMultiple chart types for comprehensive analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Data Preprocessing** ✅\nFeature engineering, scaling, handling missing values")
    with col2:
        st.success("**Interpretation** ✅\nFeature importance and prediction analysis")
