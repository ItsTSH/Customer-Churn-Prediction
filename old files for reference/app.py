import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Set page configuration
st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply dark theme
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #262730;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E1E1E;
    }
</style>
""", unsafe_allow_html=True)

# Load the model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        # Load the XGBoost model
        model = pickle.load(open('xgboost_model.pkl', 'rb'))
        
        # Load the scaler
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        
        return model, scaler
    except Exception as e:
        # If model or scaler doesn't exist, return None for demo purposes
        st.warning(f"Model or scaler file not found: {str(e)}. Using demo mode for illustration.")
        return None, None

# Function to preprocess data based on your dataset columns
def preprocess_data(df):
    # Make a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # Check for missing values
    processed_df = processed_df.dropna()
    
    # Handle categorical variables with the provided mappings
    if 'Subscription Type' in processed_df.columns:
        mapping = {'Basic': 0, 'Premium': 1, 'Standard': 2}
        processed_df['Subscription Type'] = processed_df['Subscription Type'].map(mapping)
    
    if 'Contract Length' in processed_df.columns:
        mapping = {'Annual': 0, 'Monthly': 1, 'Quarterly': 2}
        processed_df['Contract Length'] = processed_df['Contract Length'].map(mapping)
    
    # Convert Churn to binary if it's not already (0 or 1)
    if 'Churn' in processed_df.columns and not pd.api.types.is_numeric_dtype(processed_df['Churn']):
        processed_df['Churn'] = processed_df['Churn'].map({'No': 0, 'Yes': 1})
    
    return processed_df

# Function to make predictions
def predict_churn(df, model, scaler):
    if model is None:
        # If model is not available, create mock predictions for demo
        predictions = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])
        # Generate mock probability scores
        probabilities = np.random.uniform(0, 1, size=len(df))
        return predictions, probabilities
    
    # Process data for prediction
    processed_df = preprocess_data(df)
    
    # Apply scaling with the loaded scaler
    numeric_cols = ['Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 
                    'Total Spend', 'Last Interaction']
    # Get only the columns that exist in the dataframe
    cols_to_scale = [col for col in numeric_cols if col in processed_df.columns]
    
    if cols_to_scale:
        try:
            # Apply the scaler to the numeric columns
            processed_df[cols_to_scale] = scaler.transform(processed_df[cols_to_scale])
        except Exception as e:
            st.error(f"Scaling error: {str(e)}")
            # Continue without scaling if there's an error
    
    # Drop the target column if it exists
    if 'Churn' in processed_df.columns:
        processed_df = processed_df.drop('Churn', axis=1)
    
    # Convert to DMatrix before prediction
    try:
        dmatrix = xgb.DMatrix(processed_df)
        
        # Make predictions
        predictions = model.predict(dmatrix)
        probabilities = predictions  # For XGBoost binary classification, predictions are probabilities
        # Convert probabilities to binary predictions (0 or 1)
        binary_predictions = (probabilities > 0.5).astype(int)
        
        return binary_predictions, probabilities
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        # Return mock data in case of error
        predictions = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])
        probabilities = np.random.uniform(0, 1, size=len(df))
        return predictions, probabilities

# Function to categorize risk levels
def categorize_risk(probability):
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.7:
        return "Medium Risk"
    else:
        return "High Risk"

# Create sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Prediction"])

# Upload file section in sidebar
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Initialize session state if it doesn't exist
if 'data' not in st.session_state:
    st.session_state.data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'probabilities' not in st.session_state:
    st.session_state.probabilities = None
if 'risk_categories' not in st.session_state:
    st.session_state.risk_categories = None

# Load sample data if nothing is uploaded
if uploaded_file is None:
    # Create sample data that matches your dataset format
    st.sidebar.info("No file uploaded. Using sample data.")
    sample_data = {
        'Tenure': np.random.randint(1, 72, 100),
        'Usage Frequency': np.random.randint(1, 30, 100),
        'Support Calls': np.random.randint(0, 10, 100),
        'Payment Delay': np.random.randint(0, 30, 100),
        'Subscription Type': np.random.choice(['Basic', 'Premium', 'Standard'], 100),
        'Contract Length': np.random.choice(['Annual', 'Monthly', 'Quarterly'], 100),
        'Total Spend': np.random.uniform(100, 5000, 100),
        'Last Interaction': np.random.randint(1, 100, 100),
        'Churn': np.random.choice([0, 1], 100)
    }
    df = pd.DataFrame(sample_data)
    if st.session_state.data is None:
        st.session_state.data = df
else:
    # Load the data from uploaded file
    df = pd.read_csv(uploaded_file)
    st.session_state.data = df

# Load model and scaler
model, scaler = load_model_and_scaler()

# Make predictions if data exists
if st.session_state.data is not None and (st.session_state.predictions is None or uploaded_file is not None):
    predictions, probabilities = predict_churn(st.session_state.data, model, scaler)
    st.session_state.predictions = predictions
    st.session_state.probabilities = probabilities
    st.session_state.risk_categories = [categorize_risk(p) for p in probabilities]

# Dashboard Page
if page == "Dashboard":
    st.title("📊 Telecom Customer Churn Dashboard")
    
    if st.session_state.data is not None:
        # Overview section
        st.header("Data Overview")
        
        # Display dataframe
        st.dataframe(st.session_state.data, use_container_width=True)
        
        # Add data statistics
        st.subheader("Numerical Data Statistics")
        
        # Get only numeric columns
        numeric_cols = st.session_state.data.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        if numeric_cols:
            # Create 3 columns for stats
            col1, col2, col3 = st.columns(3)
            
            # Display basic stats in each column
            with col1:
                st.metric("Total Customers", len(st.session_state.data))
            
            with col2:
                # Average tenure
                if 'Tenure' in st.session_state.data.columns:
                    avg_tenure = st.session_state.data['Tenure'].mean()
                    st.metric("Average Tenure", f"{avg_tenure:.2f}")
                else:
                    st.metric("Average Tenure", "N/A")
            
            with col3:
                # Average total spend
                if 'Total Spend' in st.session_state.data.columns:
                    avg_spend = st.session_state.data['Total Spend'].mean()
                    st.metric("Average Total Spend", f"${avg_spend:.2f}")
                else:
                    st.metric("Average Total Spend", "N/A")
            
            # Create tabs for different visualizations
            tab1, tab2 = st.tabs(["📈 Distribution Charts", "📊 Relationship Charts"])
            
            with tab1:
                st.subheader("Distribution of Numerical Features")
                
                # Display histograms for numeric columns
                for i in range(0, len(numeric_cols), 2):
                    cols = st.columns(2)
                    
                    for j in range(2):
                        if i + j < len(numeric_cols):
                            with cols[j]:
                                fig = px.histogram(
                                    st.session_state.data, 
                                    x=numeric_cols[i+j],
                                    title=f"Distribution of {numeric_cols[i+j]}",
                                    color_discrete_sequence=['#00BFFF'],
                                    template="plotly_dark"
                                )
                                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.subheader("Feature Relationships")
                
                # Create correlation heatmap
                if len(numeric_cols) > 1:
                    st.write("Correlation Heatmap")
                    corr = st.session_state.data[numeric_cols].corr()
                    fig = px.imshow(
                        corr,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale='Viridis',
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display some scatter plots for relevant features
                if 'Tenure' in numeric_cols and 'Total Spend' in numeric_cols:
                    fig = px.scatter(
                        st.session_state.data,
                        x='Tenure',
                        y='Total Spend',
                        title="Tenure vs Total Spend",
                        template="plotly_dark",
                        color_discrete_sequence=['#FF4B4B']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Visualizations related to churn
                    if 'Churn' in st.session_state.data.columns:
                        # Convert to numeric if it's not already
                        if st.session_state.data['Churn'].dtype == object:
                            churn_numeric = st.session_state.data['Churn'].map({'Yes': 1, 'No': 0})
                        else:
                            churn_numeric = st.session_state.data['Churn']
                            
                        fig = px.histogram(
                            st.session_state.data,
                            x='Tenure',
                            color=churn_numeric.map({1: 'Yes', 0: 'No'}),
                            barmode='overlay',
                            title="Customer Tenure by Churn Status",
                            color_discrete_map={'Yes': 'green', 'No': 'red'},
                            template="plotly_dark"
                        )
                        fig.update_layout(
                            xaxis_title="Tenure",
                            yaxis_title="Number of Customers"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Total Spend by churn status
                        fig = px.histogram(
                            st.session_state.data,
                            x='Total Spend',
                            color=churn_numeric.map({1: 'Yes', 0: 'No'}),
                            barmode='overlay',
                            title="Total Spend by Churn Status",
                            color_discrete_map={'Yes': 'green', 'No': 'red'},
                            template="plotly_dark"
                        )
                        fig.update_layout(
                            xaxis_title="Total Spend",
                            yaxis_title="Number of Customers"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Display categorical data visualizations
            st.subheader("Categorical Features")
            
            # Get categorical columns
            cat_cols = [col for col in ['Subscription Type', 'Contract Length'] 
                        if col in st.session_state.data.columns]
            
            if cat_cols:
                cols = st.columns(min(len(cat_cols), 2))
                
                for i, col in enumerate(cat_cols[:2]):
                    with cols[i]:
                        # Map encoded values back to original categories for visualization
                        if col == 'Subscription Type':
                            mapping = {0: 'Basic', 1: 'Premium', 2: 'Standard'}
                            display_col = st.session_state.data[col].map(mapping)
                        elif col == 'Contract Length':
                            mapping = {0: 'Annual', 1: 'Monthly', 2: 'Quarterly'}
                            display_col = st.session_state.data[col].map(mapping)
                        else:
                            display_col = st.session_state.data[col]
                            
                        fig = px.bar(
                            display_col.value_counts().reset_index(),
                            x='index',
                            y=col,
                            title=f"Count of {col}",
                            template="plotly_dark",
                            color_discrete_sequence=['#00FF00', '#FF00FF', '#FFFF00']
                        )
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numerical columns found in the data.")

# Prediction Page
elif page == "Prediction":
    st.title("🔮 Churn Prediction Results")
    
    if st.session_state.data is not None and st.session_state.predictions is not None:
        # Create a copy of the original data
        display_df = st.session_state.data.copy()
        
        # Add the predictions to the dataframe
        display_df['Churn Prediction'] = st.session_state.predictions
        display_df['Churn Probability'] = st.session_state.probabilities
        display_df['Risk Category'] = st.session_state.risk_categories
        
        # Display overall prediction statistics
        st.header("Prediction Overview")
        
        # Create metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            churn_count = sum(st.session_state.predictions)
            churn_percent = (churn_count / len(display_df)) * 100
            st.metric("Predicted to Churn", f"{churn_count} ({churn_percent:.1f}%)")
        
        with col2:
            high_risk = display_df[display_df['Risk Category'] == 'High Risk'].shape[0]
            high_risk_percent = (high_risk / len(display_df)) * 100
            st.metric("High Risk Customers", f"{high_risk} ({high_risk_percent:.1f}%)")
        
        with col3:
            avg_prob = np.mean(st.session_state.probabilities) * 100
            st.metric("Average Churn Probability", f"{avg_prob:.1f}%")
        
        # Create 3D pie chart of risk categories
        st.subheader("Risk Category Distribution")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            risk_counts = pd.Series(st.session_state.risk_categories).value_counts().reset_index()
            risk_counts.columns = ['Risk Category', 'Count']
            
            # Create 3D pie chart
            fig = go.Figure(data=[go.Pie(
                labels=risk_counts['Risk Category'],
                values=risk_counts['Count'],
                hole=0.3,
                textinfo='label+percent',
                textposition='inside',
                insidetextorientation='radial',
                pull=[0.1 if cat == 'High Risk' else 0 for cat in risk_counts['Risk Category']],
                marker=dict(colors=['#66BB6A', '#FFA726', '#EF5350'])
            )])
            
            fig.update_layout(
                title_text="Distribution of Risk Categories",
                template="plotly_dark",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                scene=dict(aspectmode="cube"),
                margin=dict(t=50, b=0, l=0, r=0),
                showlegend=False
            )
            
            # Update to make it 3D-like
            fig.update_traces(
                rotation=45,
                textfont=dict(size=12, color="white"),
                marker=dict(line=dict(color='#000000', width=1.5))
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Display risk category counts in a table
            st.write("Risk Category Counts")
            st.dataframe(risk_counts, use_container_width=True)
            
            # Add a gauge chart for overall churn risk
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_prob,
                title={'text': "Overall Churn Risk"},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#FF4136"},
                    'steps': [
                        {'range': [0, 30], 'color': '#66BB6A'},
                        {'range': [30, 70], 'color': '#FFA726'},
                        {'range': [70, 100], 'color': '#EF5350'}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': avg_prob
                    }
                }
            ))
            
            fig.update_layout(
                template="plotly_dark",
                margin=dict(t=30, b=0, l=30, r=30)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Display data with prediction results
        st.subheader("Detailed Prediction Results")
        
        # Create result dataframe with key columns
        result_columns = ['Tenure', 'Usage Frequency', 'Support Calls', 'Total Spend']
        result_columns = [col for col in result_columns if col in display_df.columns]
        result_columns.extend(['Churn Probability', 'Risk Category', 'Churn Prediction'])
        
        # Ensure we have some columns
        if not result_columns:
            result_columns = display_df.columns.tolist()[-5:] + ['Churn Probability', 'Risk Category', 'Churn Prediction']
        
        results_df = display_df[result_columns].copy()
        
        st.dataframe(results_df.sort_values(by='Churn Probability', ascending=False), use_container_width=True)
        
        # Add download button for predictions
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download Prediction Results",
            data=csv,
            file_name="telecom_churn_predictions.csv",
            mime="text/csv"
        )
        
        # Display additional visualizations
        st.subheader("Prediction Analysis")
        
        # Create tabs for different analyses
        tab1, tab2 = st.tabs(["📊 Feature Analysis", "🔍 High Risk Analysis"])
        
        with tab1:
            # Analyze tenure and total spend by risk category
            if 'Tenure' in display_df.columns:
                fig = px.box(
                    display_df, 
                    x='Risk Category', 
                    y='Tenure', 
                    color='Risk Category',
                    title='Tenure by Risk Category',
                    template="plotly_dark",
                    color_discrete_map={
                        'Low Risk': '#66BB6A',
                        'Medium Risk': '#FFA726',
                        'High Risk': '#EF5350'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            if 'Total Spend' in display_df.columns:
                fig = px.box(
                    display_df, 
                    x='Risk Category', 
                    y='Total Spend', 
                    color='Risk Category',
                    title='Total Spend by Risk Category',
                    template="plotly_dark",
                    color_discrete_map={
                        'Low Risk': '#66BB6A',
                        'Medium Risk': '#FFA726',
                        'High Risk': '#EF5350'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Analyze categorical features if they exist
            if 'Contract Length' in display_df.columns:
                # Map numeric values back to categories for display
                mapping = {0: 'Annual', 1: 'Monthly', 2: 'Quarterly'}
                display_df['Contract Length Display'] = display_df['Contract Length'].map(mapping)
                
                # Calculate the percentage of high risk customers by contract type
                contract_risk = pd.crosstab(
                    display_df['Contract Length Display'], 
                    display_df['Risk Category'],
                    normalize='index'
                ).reset_index()
                
                # Melt the dataframe for easier plotting
                contract_risk_melted = pd.melt(
                    contract_risk, 
                    id_vars=['Contract Length Display'], 
                    value_vars=['Low Risk', 'Medium Risk', 'High Risk'],
                    var_name='Risk Category',
                    value_name='Percentage'
                )
                
                # Multiply by 100 to get percentage
                contract_risk_melted['Percentage'] = contract_risk_melted['Percentage'] * 100
                
                fig = px.bar(
                    contract_risk_melted,
                    x='Contract Length Display',
                    y='Percentage',
                    color='Risk Category',
                    title='Risk Category Distribution by Contract Length',
                    template="plotly_dark",
                    color_discrete_map={
                        'Low Risk': '#66BB6A',
                        'Medium Risk': '#FFA726',
                        'High Risk': '#EF5350'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Analyze high risk customers
            high_risk_df = display_df[display_df['Risk Category'] == 'High Risk']
            
            if not high_risk_df.empty:
                st.write(f"Number of High Risk Customers: {len(high_risk_df)}")
                
                # Create a summary of key characteristics
                st.write("Key Characteristics of High Risk Customers:")
                
                # Tenure analysis if it exists
                if 'Tenure' in high_risk_df.columns:
                    col1, col2 = st.columns(2)
                    with col1:
                        avg_tenure_high = high_risk_df['Tenure'].mean()
                        avg_tenure_all = display_df['Tenure'].mean()
                        st.metric(
                            "Average Tenure of High Risk Customers", 
                            f"{avg_tenure_high:.1f}",
                            f"{avg_tenure_high - avg_tenure_all:.1f} vs. overall"
                        )
                    
                    with col2:
                        if 'Total Spend' in high_risk_df.columns:
                            avg_spend_high = high_risk_df['Total Spend'].mean()
                            avg_spend_all = display_df['Total Spend'].mean()
                            st.metric(
                                "Average Total Spend of High Risk", 
                                f"${avg_spend_high:.2f}",
                                f"${avg_spend_high - avg_spend_all:.2f} vs. overall"
                            )
                
                # Check if specific columns exist for analysis
                cat_cols_to_analyze = ['Subscription Type', 'Contract Length']
                cat_cols_to_analyze = [col for col in cat_cols_to_analyze if col in high_risk_df.columns]
                
                if cat_cols_to_analyze:
                    st.write("Distribution of High Risk Customers by Key Categories:")
                    
                    for col in cat_cols_to_analyze:
                        # Map numeric values back to categories for display
                        if col == 'Subscription Type':
                            mapping = {0: 'Basic', 1: 'Premium', 2: 'Standard'}
                            high_risk_df[f'{col} Display'] = high_risk_df[col].map(mapping)
                            display_col = f'{col} Display'
                        elif col == 'Contract Length':
                            mapping = {0: 'Annual', 1: 'Monthly', 2: 'Quarterly'}
                            high_risk_df[f'{col} Display'] = high_risk_df[col].map(mapping)
                            display_col = f'{col} Display'
                        else:
                            display_col = col
                        
                        # Calculate distribution
                        category_dist = high_risk_df[display_col].value_counts(normalize=True) * 100
                        
                        fig = px.pie(
                            names=category_dist.index,
                            values=category_dist.values,
                            title=f"Distribution of High Risk Customers by {col}",
                            template="plotly_dark",
                            hole=0.3
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No high risk customers identified in the dataset.")
    else:
        st.info("Please upload data to see prediction results.")

# Add a footer
st.markdown("---")
st.markdown("Telecom Churn Prediction App © 2025")