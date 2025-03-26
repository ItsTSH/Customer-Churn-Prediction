import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import os

def generate_sample_data():
    """
    Generate sample data with exactly the specified columns
    """
    np.random.seed(42)
    sample_data = {
        'Tenure': np.random.randint(1, 72, 100),
        'Usage Frequency': np.random.randint(1, 30, 100),
        'Support Calls': np.random.randint(0, 10, 100),
        'Payment Delay': np.random.randint(0, 30, 100),
        'Subscription Type': np.random.choice(['Basic', 'Premium', 'Standard'], 100),
        'Contract Length': np.random.choice(['Annual', 'Monthly', 'Quarterly'], 100),
        'Total Spend': np.random.uniform(100, 5000, 100),
        'Last Interaction': np.random.randint(1, 365, 100),
        'Churn': np.random.choice([0, 1], 100)
    }
    return pd.DataFrame(sample_data)

def validate_dataframe(data):
    """
    Validate that the dataframe has exactly the required columns
    """
    required_columns = [
        'Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 
        'Subscription Type', 'Contract Length', 'Total Spend', 
        'Last Interaction', 'Churn'
    ]
    
    # Check for missing columns
    missing_columns = set(required_columns) - set(data.columns)
    if missing_columns:
        st.error(f"Missing columns: {', '.join(missing_columns)}")
        return False
    
    # Check for extra columns
    extra_columns = set(data.columns) - set(required_columns)
    if extra_columns:
        st.warning(f"Extra columns will be dropped: {', '.join(extra_columns)}")
        data = data[required_columns]
    
    return data

def advanced_statistical_analysis(data):
    """
    Perform advanced statistical analysis on the dataset
    """
    st.header("🔬 Advanced Statistical Analysis")
    
    # Descriptive Statistics
    st.subheader("Comprehensive Descriptive Statistics")
    numeric_cols = ['Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']
    
    # Detailed descriptive stats
    desc_stats = data[numeric_cols].describe()
    
    # Add skewness and kurtosis
    desc_stats.loc['skewness'] = data[numeric_cols].apply(lambda x: x.skew())
    desc_stats.loc['kurtosis'] = data[numeric_cols].apply(lambda x: x.kurtosis())
    
    st.dataframe(desc_stats.T, use_container_width=True)
    
    # Outlier Detection
    st.subheader("Outlier Analysis")
    outlier_cols = st.columns(3)
    
    for i, col in enumerate(numeric_cols[:3]):
        with outlier_cols[i]:
            # Box plot to show outliers
            fig = px.box(
                data, 
                y=col, 
                title=f"Outliers in {col}",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Categorical Feature Analysis
    st.subheader("Categorical Feature Distribution")
    cat_cols = ['Subscription Type', 'Contract Length']
    
    cols = st.columns(2)
    for i, col in enumerate(cat_cols):
        with cols[i]:
            # Pie chart for categorical variables
            fig = px.pie(
                data, 
                names=col, 
                title=f"Distribution of {col}",
                hole=0.3,
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Churn Analysis
    st.subheader("Churn Analysis")
    
    # Churn by Subscription Type
    churn_by_sub = data.groupby(['Subscription Type', 'Churn']).size().unstack(fill_value=0)
    churn_by_sub_percent = churn_by_sub.div(churn_by_sub.sum(axis=1), axis=0) * 100
    
    fig = px.bar(
        x=churn_by_sub_percent.index, 
        y=churn_by_sub_percent[1],
        title="Churn Percentage by Subscription Type",
        labels={'x': 'Subscription Type', 'y': 'Churn Percentage'},
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Churn by Contract Length
    churn_by_contract = data.groupby(['Contract Length', 'Churn']).size().unstack(fill_value=0)
    churn_by_contract_percent = churn_by_contract.div(churn_by_contract.sum(axis=1), axis=0) * 100
    
    fig = px.bar(
        x=churn_by_contract_percent.index, 
        y=churn_by_contract_percent[1],
        title="Churn Percentage by Contract Length",
        labels={'x': 'Contract Length', 'y': 'Churn Percentage'},
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

def create_comprehensive_dashboard(data):
    # -----Display Dataset Overview-----
    st.header("📊 Dataset Overview")
    
    # Display full dataset
    st.subheader("Full Dataset")
    st.dataframe(data, use_container_width=True)
    
    # Basic dataset information
    st.subheader("Dataset Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", len(data))
    
    with col2:
        st.metric("Total Columns", len(data.columns))
    
    with col3:
        churn_rate = data['Churn'].mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.2f}%")
    
    # Descriptive statistics
    st.subheader("Descriptive Statistics")
    numeric_cols = ['Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']
    st.dataframe(data[numeric_cols].describe(), use_container_width=True)

    # -----Display Simple Data Analytics-----
    st.header("📈 Simple Data Visualizations")
    
    # Categorical Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Subscription Type Distribution")
        fig_sub = px.pie(
            data, 
            names='Subscription Type', 
            title='Customer Subscription Types',
            hole=0.3,
            template="plotly_white"
        )
        st.plotly_chart(fig_sub, use_container_width=True)
    
    with col2:
        st.subheader("Contract Length Distribution")
        fig_contract = px.pie(
            data, 
            names='Contract Length', 
            title='Contract Length Types',
            hole=0.3,
            template="plotly_white"
        )
        st.plotly_chart(fig_contract, use_container_width=True)
    
    # Churn Analysis
    st.subheader("Churn Analysis")
    
    # Churn by Subscription Type
    churn_by_sub = data.groupby(['Subscription Type', 'Churn']).size().unstack(fill_value=0)
    churn_by_sub_percent = churn_by_sub.div(churn_by_sub.sum(axis=1), axis=0) * 100
    
    fig_churn_sub = px.bar(
        x=churn_by_sub_percent.index, 
        y=churn_by_sub_percent[1],
        title="Churn Percentage by Subscription Type",
        labels={'x': 'Subscription Type', 'y': 'Churn Percentage'},
        template="plotly_white"
    )
    st.plotly_chart(fig_churn_sub, use_container_width=True)
    
    # Numeric Feature Distributions
    st.subheader("Numeric Feature Distributions")
    numeric_cols = ['Tenure', 'Usage Frequency', 'Total Spend']
    
    cols = st.columns(3)
    for i, col in enumerate(numeric_cols):
        with cols[i]:
            fig = px.histogram(
                data, 
                x=col, 
                title=f'Distribution of {col}',
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)


    # ----- Display Advanced Data Analytics-----
    st.title("🔍 Advanced Data Analytics")
    
    # Tabs for different analysis types
    tabs = [
        "Overview", 
        "Descriptive Analysis", 
        "Statistical Analysis", 
        "Predictive Insights"
    ]
    
    tab1, tab2, tab3, tab4 = st.tabs(tabs)
    
    with tab1:
        st.header("Data Overview")
        
        # Basic data info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", len(data))
        
        with col2:
            st.metric("Columns", len(data.columns))
        
        with col3:
            churn_rate = data['Churn'].mean() * 100
            st.metric("Churn Rate", f"{churn_rate:.2f}%")
        
        # Correlation Heatmap
        st.subheader("Feature Correlations")
        numeric_cols = ['Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']
        corr_matrix = data[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix, 
            text_auto=True, 
            title="Feature Correlation Heatmap",
            template="plotly_dark"
        )
        # Increase figure size
        fig.update_layout(
            width=800,  # Increase width
            height=600   # Increase height
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Detailed Descriptive Analysis")
        
        # Key numeric feature distributions
        numeric_cols = ['Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']
        
        for col in numeric_cols:
            st.subheader(f"Distribution of {col}")
            
            # Histogram with kernel density estimate
            fig = px.histogram(
                data, 
                x=col, 
                marginal='violin',
                title=f"Distribution of {col}",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Call the advanced statistical analysis function
        advanced_statistical_analysis(data)
    
    with tab4:
        st.header("Predictive Insights")
        
        # Feature Importance Estimation
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        
        # Prepare data for ML
        ml_data = data.copy()
        
        # Encode categorical variables
        le = LabelEncoder()
        cat_cols = ['Subscription Type', 'Contract Length']
        for col in cat_cols:
            ml_data[col] = le.fit_transform(ml_data[col])
        
        # Separate features and target
        X = ml_data.drop('Churn', axis=1)
        y = ml_data['Churn']
        
        # Train Random Forest for feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importances
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Plot feature importances
        fig = px.bar(
            feature_importance, 
            x='feature', 
            y='importance',
            title="Feature Importance for Churn Prediction",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)


def main():
    # File uploader
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    
    try:
        # Get the absolute path of the dataset
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_path = os.path.join(base_dir, 'dataset', 'dataset.csv')
        local_dataset = pd.read_csv(dataset_path) if os.path.exists(dataset_path) else None
    except Exception as e:
        st.sidebar.error(f"Error loading local dataset: {e}")
        local_dataset = None
    
    # Determine which dataset to use
    if uploaded_file is not None:
        # Use uploaded dataset
        data = pd.read_csv(uploaded_file)
    elif local_dataset is not None:
        # Use local dataset
        data = local_dataset
    else:
        # Generate sample dataset
        data = generate_sample_data()
    
    # Validate and potentially modify the dataframe
    validated_data = validate_dataframe(data)
    
    if validated_data is not False:
        create_comprehensive_dashboard(validated_data)
    else:
        st.error("Please provide a dataset with the correct columns.")

if __name__ == "__main__":
    main()