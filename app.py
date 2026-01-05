import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Earnings Manipulation Detector",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS ==========
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E3A8A;
        margin: 10px 0;
    }
    .prediction-box {
        padding: 25px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
        margin: 20px 0;
    }
    .safe {
        background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%);
        color: #065F46;
        border: 2px solid #34D399;
    }
    .risk {
        background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
        color: #991B1B;
        border: 2px solid #F87171;
    }
    .stButton button {
        width: 100%;
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        font-weight: bold;
        border: none;
    }
    .feature-input {
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ========== LOAD MODEL ==========
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except:
        st.error("⚠️ Model not found. Please run `python train_model.py` first.")
        return None, None

model, scaler = load_model()
features = ['DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'ACCR', 'LEVI']

# ========== SIDEBAR ==========
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
    st.title("Earnings Detector")
    st.markdown("---")
    
    mode = st.radio(
        "SELECT MODE:",
        ["🏠 Home", "🔍 Single Analysis", "📁 Batch Analysis", "📊 Model Info"]
    )
    
    st.markdown("---")
    with st.expander("📚 How to Use"):
        st.markdown("""
        1. **Run model training** first
        2. **Single Analysis**: Input ratios manually
        3. **Batch Analysis**: Upload Excel/CSV file
        4. **Model Info**: View performance metrics
        """)
    
    if model:
        st.success("✅ Model Loaded")
    else:
        st.warning("⚠️ Model Not Loaded")

# ========== MAIN CONTENT ==========
if mode == "🏠 Home":
    st.markdown('<h1 class="main-header">📊 Earnings Manipulation Detection System</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="card">
        <h3>🚀 Quick Start</h3>
        <ol>
        <li><b>Install requirements</b>: pip install -r requirements.txt</li>
        <li><b>Train model</b>: python train_model.py</li>
        <li><b>Run app</b>: streamlit run app.py</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
        <h3>📈 What We Detect</h3>
        <ul>
        <li>Revenue manipulation</li>
        <li>Expense manipulation</li>
        <li>Asset quality issues</li>
        <li>Accrual anomalies</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
        <h3>🔍 Features Used</h3>
        <ul>
        <li><b>DSRI</b>: Days Sales in Receivables Index</li>
        <li><b>GMI</b>: Gross Margin Index</li>
        <li><b>AQI</b>: Asset Quality Index</li>
        <li><b>SGI</b>: Sales Growth Index</li>
        <li><b>DEPI</b>: Depreciation Index</li>
        <li><b>SGAI</b>: SG&A Expense Index</li>
        <li><b>ACCR</b>: Accruals</li>
        <li><b>LEVI</b>: Leverage Index</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("💡 **Tip**: Place your 'Earnings Manipulator (1).xlsx' file in the same folder as app.py")

elif mode == "🔍 Single Analysis":
    st.title("🔍 Single Company Analysis")
    
    if model is None:
        st.error("Model not loaded. Please run train_model.py first.")
        st.stop()
    
    # Create input form
    with st.form("prediction_form"):
        st.subheader("Enter Financial Ratios")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="feature-input">', unsafe_allow_html=True)
            dsri = st.number_input("**DSRI** (Days' Sales in Receivables)", 
                                 value=1.0, min_value=0.0, max_value=50.0, step=0.1,
                                 help=">1.4 suggests aggressive revenue recognition")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feature-input">', unsafe_allow_html=True)
            gmi = st.number_input("**GMI** (Gross Margin Index)", 
                                value=1.0, min_value=-50.0, max_value=50.0, step=0.1,
                                help=">1.1 indicates deteriorating margins")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feature-input">', unsafe_allow_html=True)
            aqi = st.number_input("**AQI** (Asset Quality Index)", 
                                value=1.0, min_value=-50.0, max_value=100.0, step=0.1,
                                help=">1.2 suggests decreasing asset quality")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feature-input">', unsafe_allow_html=True)
            sgi = st.number_input("**SGI** (Sales Growth Index)", 
                                value=1.0, min_value=0.0, max_value=50.0, step=0.1,
                                help="Rapid growth may indicate manipulation")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="feature-input">', unsafe_allow_html=True)
            depi = st.number_input("**DEPI** (Depreciation Index)", 
                                 value=1.0, min_value=0.0, max_value=10.0, step=0.1,
                                 help=">1 suggests slower depreciation")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feature-input">', unsafe_allow_html=True)
            sgai = st.number_input("**SGAI** (SG&A Expense Index)", 
                                 value=1.0, min_value=0.0, max_value=50.0, step=0.1,
                                 help="<1 may indicate expense manipulation")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feature-input">', unsafe_allow_html=True)
            accr = st.number_input("**ACCR** (Accruals)", 
                                 value=0.0, min_value=-5.0, max_value=5.0, step=0.01,
                                 help="Positive values suggest earnings management")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="feature-input">', unsafe_allow_html=True)
            levi = st.number_input("**LEVI** (Leverage Index)", 
                                 value=1.0, min_value=0.0, max_value=50.0, step=0.1,
                                 help=">1 indicates increasing leverage")
            st.markdown('</div>', unsafe_allow_html=True)
        
        submit = st.form_submit_button("🔍 PREDICT MANIPULATION RISK", type="primary", use_container_width=True)
    
    if submit:
        # Prepare input
        input_data = np.array([[dsri, gmi, aqi, sgi, depi, sgai, accr, levi]])
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        risk_score = probabilities[1] * 100
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        col_res1, col_res2 = st.columns([1, 1])
        
        with col_res1:
            if prediction == 1:
                st.markdown(f'<div class="prediction-box risk">⚠️ HIGH RISK<br>Probability: {risk_score:.1f}%</div>', 
                          unsafe_allow_html=True)
                st.error("**Alert**: High probability of earnings manipulation detected.")
            else:
                st.markdown(f'<div class="prediction-box safe">✅ LOW RISK<br>Probability: {100-risk_score:.1f}%</div>', 
                          unsafe_allow_html=True)
                st.success("**Clear**: No significant manipulation indicators found.")
        
        with col_res2:
            # Risk gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                title={'text': "Risk Score", 'font': {'size': 20}},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "green"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': risk_score
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed probabilities
        st.subheader("📈 Detailed Analysis")
        prob_col1, prob_col2 = st.columns(2)
        
        with prob_col1:
            st.metric("No Manipulation", f"{probabilities[0]*100:.1f}%")
        with prob_col2:
            st.metric("Potential Manipulation", f"{probabilities[1]*100:.1f}%")

elif mode == "📁 Batch Analysis":
    st.title("📁 Batch Company Analysis")
    
    if model is None:
        st.error("Model not loaded. Please run train_model.py first.")
        st.stop()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Excel or CSV file with company data", 
        type=['xlsx', 'csv'],
        help="File should contain columns: DSRI, GMI, AQI, SGI, DEPI, SGAI, ACCR, LEVI"
    )
    
    if uploaded_file is not None:
        try:
            # Read file
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File loaded: {len(df)} records")
            
            # Show preview
            with st.expander("Preview data"):
                st.dataframe(df.head())
            
            # Check required columns
            missing_cols = [f for f in features if f not in df.columns]
            
            if missing_cols:
                st.error(f"Missing columns: {', '.join(missing_cols)}")
                st.info(f"Required columns: {', '.join(features)}")
            else:
                if st.button("🚀 ANALYZE ALL COMPANIES", type="primary", use_container_width=True):
                    # Process all companies
                    X = df[features]
                    X_scaled = scaler.transform(X)
                    
                    predictions = model.predict(X_scaled)
                    probabilities = model.predict_proba(X_scaled)[:, 1]
                    
                    # Create results dataframe
                    results = df.copy()
                    results['Prediction'] = ['Yes' if p == 1 else 'No' for p in predictions]
                    results['Risk_Score'] = probabilities
                    results['Risk_Level'] = ['High' if prob > 0.5 else 'Low' for prob in probabilities]
                    
                    # Summary statistics
                    total = len(results)
                    high_risk = sum(predictions == 1)
                    low_risk = sum(predictions == 0)
                    
                    st.subheader("📊 Summary")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Companies", total)
                    with col2:
                        st.metric("High Risk", high_risk, f"{high_risk/total*100:.1f}%")
                    with col3:
                        st.metric("Low Risk", low_risk, f"{low_risk/total*100:.1f}%")
                    with col4:
                        avg_risk = probabilities.mean() * 100
                        st.metric("Avg Risk", f"{avg_risk:.1f}%")
                    
                    # Results table
                    st.subheader("📋 Detailed Results")
                    st.dataframe(results[['Company ID', 'Prediction', 'Risk_Score', 'Risk_Level'] + features])
                    
                    # Visualizations
                    col_viz1, col_viz2 = st.columns(2)
                    
                    with col_viz1:
                        # Pie chart
                        fig1 = px.pie(
                            names=['Low Risk', 'High Risk'],
                            values=[low_risk, high_risk],
                            title="Risk Distribution",
                            color=['Low Risk', 'High Risk'],
                            color_discrete_map={'Low Risk': 'green', 'High Risk': 'red'}
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col_viz2:
                        # Histogram of risk scores
                        fig2 = px.histogram(
                            x=probabilities,
                            nbins=20,
                            title="Risk Score Distribution",
                            labels={'x': 'Risk Score', 'y': 'Count'},
                            color_discrete_sequence=['#FF6B6B']
                        )
                        fig2.add_vline(x=0.5, line_dash="dash", line_color="red")
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    # Download button
                    csv = results.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Full Results (CSV)",
                        data=csv,
                        file_name="earnings_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        # Show template download
        st.info("💡 Need a template?")
        template_data = pd.DataFrame(columns=['Company ID'] + features)
        csv = template_data.to_csv(index=False)
        st.download_button(
            label="📄 Download Template (CSV)",
            data=csv,
            file_name="template.csv",
            mime="text/csv"
        )

elif mode == "📊 Model Info":
    st.title("📊 Model Information")
    
    if model is None:
        st.error("Model not loaded. Please run train_model.py first.")
        st.stop()
    
    # Load sample data for statistics
    try:
        df = pd.read_excel('Earnings Manipulator (1).xlsx', sheet_name='Data for Model Development')
        
        # Feature descriptions
        feature_descriptions = {
            'DSRI': "Days' Sales in Receivables Index - Measures the change in receivables relative to sales",
            'GMI': "Gross Margin Index - Measures gross margin deterioration from prior period",
            'AQI': "Asset Quality Index - Measures the change in asset composition quality",
            'SGI': "Sales Growth Index - Measures the rate of sales growth",
            'DEPI': "Depreciation Index - Measures the change in depreciation rates",
            'SGAI': "SG&A Expense Index - Measures the change in selling, general and administrative expenses",
            'ACCR': "Accruals - Measures total accruals to total assets",
            'LEVI': "Leverage Index - Measures the change in financial leverage"
        }
        
        # Model coefficients
        coefficients = pd.DataFrame({
            'Feature': features,
            'Coefficient': model.coef_[0],
            'Importance': np.abs(model.coef_[0])
        }).sort_values('Importance', ascending=False)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📈 Feature Importance")
            fig = px.bar(
                coefficients,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Feature Impact on Prediction",
                color='Coefficient',
                color_continuous_scale='RdBu'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📊 Dataset Statistics")
            st.write(f"Total samples: {len(df)}")
            st.write(f"Manipulators: {sum(df['Manipulator'] == 'Yes')}")
            st.write(f"Non-manipulators: {sum(df['Manipulator'] == 'No')}")
        
        with col2:
            st.subheader("🔍 Feature Details")
            for feature in features:
                with st.expander(f"**{feature}**: {feature_descriptions[feature].split(' - ')[0]}"):
                    st.write(feature_descriptions[feature])
                    st.write(f"Mean: {df[feature].mean():.3f}")
                    st.write(f"Std Dev: {df[feature].std():.3f}")
                    st.write(f"Min: {df[feature].min():.3f}")
                    st.write(f"Max: {df[feature].max():.3f}")
        
        # Interpretation guidelines
        st.subheader("📋 Interpretation Guidelines")
        st.markdown("""
        | Ratio | Normal Range | Warning Signal |
        |-------|--------------|----------------|
        | **DSRI** | < 1.4 | > 1.4 suggests aggressive revenue recognition |
        | **GMI** | < 1.1 | > 1.1 indicates deteriorating margins |
        | **AQI** | < 1.2 | > 1.2 suggests decreasing asset quality |
        | **SGI** | 0.8 - 1.2 | Extreme values may indicate manipulation |
        | **DEPI** | 0.9 - 1.1 | > 1.1 suggests slower depreciation |
        | **SGAI** | 0.9 - 1.1 | < 0.9 may indicate expense manipulation |
        | **ACCR** | -0.1 to 0.1 | High positive values suggest earnings management |
        | **LEVI** | 0.8 - 1.2 | > 1.2 indicates increasing leverage risk |
        """)
    
    except:
        st.info("Load dataset to see feature statistics")

# ========== FOOTER ==========
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9rem;'>
    Earnings Manipulation Detection System v1.0 | Built with Streamlit | For Academic Use
    </div>
    """,
    unsafe_allow_html=True
)
