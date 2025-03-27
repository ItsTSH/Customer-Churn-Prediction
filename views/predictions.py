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
import os
import shap
import seaborn as sns
from scipy import stats

# Load the model and scaler
@st.cache_resource
def load_model():
    try:
        # Get the base directory (Customer-Churn-master)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Construct absolute paths for model
        model_path = os.path.join(base_dir, 'model', 'xgboost_model.pkl')

        # Load the XGBoost model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load the scaler
        #with open(scaler_path, 'rb') as f:
         #   scaler = pickle.load(f)
        
        return model #, scaler
    except Exception as e:
        # If model doesn't exist, return None for demo purposes
        st.warning(f"Model file not found: {str(e)}. Using demo mode for illustration.")
        return None, None
    
def generate_sample_data():
    # -----Generate Sample Data (exception handling)-----
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
    # -----Function to Validate the DataFrame for the specified columns-----
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

def preprocess_data(df):
    # -----Preprocess data based on the dataset columns-----
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

def data_scaler(processed_df):
    # -----Manual Scaling because scaler.pkl file has some issues currently-----
    scaled_df = processed_df
    scaler = StandardScaler()
    scaled_df[['Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']] = scaler.fit_transform(scaled_df[['Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 'Total Spend', 'Last Interaction']])

    return scaled_df

def predict_churn(df, model):
    # -----Make Predictions-----
    if model is None:
        # If model is not available, create mock predictions for demo
        predictions = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])
        # Generate mock probability scores
        probabilities = np.random.uniform(0, 1, size=len(df))
        return predictions, probabilities
    
    # Process data for prediction
    processed_df = preprocess_data(df)
    
    # Apply scaling with the loaded scaler
    #numeric_cols = ['Tenure', 'Usage Frequency', 'Support Calls', 'Payment Delay', 
     #               'Total Spend', 'Last Interaction']
    # Get only the columns that exist in the dataframe
    #cols_to_scale = [col for col in numeric_cols if col in processed_df.columns]
    
    #if cols_to_scale:
    try:
        # Apply the scaler to the numeric columns
        scaled_data = data_scaler(processed_df)
    except Exception as e:
        st.error(f"Scaling error: {str(e)}")
        # Continue without scaling if there's an error
    
    # Drop the target column if it exists
    if 'Churn' in scaled_data.columns:
        scaled_data = scaled_data.drop('Churn', axis=1)
    
    # Convert to DMatrix before prediction
    try:
        dmatrix = xgb.DMatrix(scaled_data)
        
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


def categorize_risk(probability):
    # -----Categorize based on probability of churn-----
    if probability < 0.33:
        return "Low Risk"
    elif probability < 0.66:
        return "Medium Risk"
    else:
        return "High Risk"

def plot_feature_importance(model, processed_df):
    """
    Plot feature importance using both built-in XGBoost importance and SHAP values
    """
    st.subheader("Feature Importance Analysis")
    
    # Prepare the data for feature importance
    X = processed_df.drop('Churn', axis=1)
    
    # XGBoost Built-in Feature Importance
    xgb_importance = model.get_score(importance_type='weight')
    importance_df = pd.DataFrame.from_dict(xgb_importance, orient='index', columns=['Importance'])
    importance_df.index.name = 'Feature'
    importance_df = importance_df.reset_index().sort_values('Importance', ascending=False)
    
    # Plot XGBoost Built-in Feature Importance
    fig1 = px.bar(
        importance_df, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        title='XGBoost Built-in Feature Importance',
        template="plotly_dark"
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    # SHAP Values Analysis (if possible)
    try:
        # Convert data to DMatrix
        dmatrix = xgb.DMatrix(X)
        
        # Compute SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Plot SHAP summary plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.title("SHAP Feature Importance")
        st.pyplot(fig)
        
        # Detailed SHAP analysis
        st.subheader("SHAP Feature Impact Analysis")
        
        # Compute mean absolute SHAP values
        shap_importance = np.abs(shap_values).mean(axis=0)
        shap_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Mean Absolute SHAP Value': shap_importance
        }).sort_values('Mean Absolute SHAP Value', ascending=False)
        
        # Plot SHAP importance
        fig2 = px.bar(
            shap_importance_df, 
            x='Mean Absolute SHAP Value', 
            y='Feature', 
            orientation='h',
            title='SHAP Feature Impact',
            template="plotly_dark"
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        return shap_importance_df
    
    except Exception as e:
        st.warning(f"SHAP analysis encountered an issue: {e}")
        return None

# Advanced churn analysis
def advanced_churn_analysis(display_df):
    """
    Perform a quick and optimized churn analysis to provide business insights
    """
    st.subheader("🔬 Advanced Churn Insights")
    
    # Focus on high-risk customers
    high_risk_df = display_df[display_df['Risk Category'] == 'High Risk'].copy()
    
    if high_risk_df.empty:
        st.info("No high-risk customers found for detailed analysis.")
        return
    
    # Comparative Analysis Section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Correlation between Support Calls and Churn Risk
        if 'Support Calls' in display_df.columns:
            support_correlation = display_df['Support Calls'].corr(display_df['Churn Probability'])
            st.metric(
                "Support Calls & Churn Correlation", 
                f"{support_correlation:.2f}",
                help="Positive values indicate more support calls relate to higher churn risk"
            )
    
    with col2:
        # Payment Delay Impact
        if 'Payment Delay' in display_df.columns:
            payment_correlation = display_df['Payment Delay'].corr(display_df['Churn Probability'])
            st.metric(
                "Payment Delay & Churn Correlation", 
                f"{payment_correlation:.2f}",
                help="Positive values suggest payment delays increase churn risk"
            )
    
    with col3:
        # Usage Frequency Impact
        if 'Usage Frequency' in display_df.columns:
            usage_correlation = display_df['Usage Frequency'].corr(display_df['Churn Probability'])
            st.metric(
                "Usage Frequency & Churn Correlation", 
                f"{usage_correlation:.2f}",
                help="Negative values might indicate low engagement leads to churn"
            )
    
    # Key Insights Generation
    st.subheader("Top Churn Risk Drivers")
    
    # Prepare comparative analysis
    insights = []
    
    # Support Calls Analysis
    if 'Support Calls' in display_df.columns:
        avg_support_calls_high_risk = high_risk_df['Support Calls'].mean()
        avg_support_calls_overall = display_df['Support Calls'].mean()
        
        if avg_support_calls_high_risk > avg_support_calls_overall * 1.5:
            insights.append({
                'category': 'Support Calls',
                'impact': 'High',
                'description': f"High-risk customers average {avg_support_calls_high_risk:.1f} support calls vs. {avg_support_calls_overall:.1f} overall."
            })
    
    # Payment Delay Analysis
    if 'Payment Delay' in display_df.columns:
        avg_payment_delay_high_risk = high_risk_df['Payment Delay'].mean()
        avg_payment_delay_overall = display_df['Payment Delay'].mean()
        
        if avg_payment_delay_high_risk > avg_payment_delay_overall * 1.5:
            insights.append({
                'category': 'Payment Delay',
                'impact': 'High',
                'description': f"High-risk customers have an average payment delay of {avg_payment_delay_high_risk:.1f} days vs. {avg_payment_delay_overall:.1f} days overall."
            })
    
    # Total Spend Analysis
    if 'Total Spend' in display_df.columns:
        avg_spend_high_risk = high_risk_df['Total Spend'].mean()
        avg_spend_overall = display_df['Total Spend'].mean()
        
        if avg_spend_high_risk < avg_spend_overall * 0.6:
            insights.append({
                'category': 'Total Spend',
                'impact': 'Medium',
                'description': f"High-risk customers spend ${avg_spend_high_risk:.2f} vs. ${avg_spend_overall:.2f} overall."
            })
    
    # Usage Frequency Analysis
    if 'Usage Frequency' in display_df.columns:
        avg_usage_high_risk = high_risk_df['Usage Frequency'].mean()
        avg_usage_overall = display_df['Usage Frequency'].mean()
        
        if avg_usage_high_risk < avg_usage_overall * 0.6:
            insights.append({
                'category': 'Usage Frequency',
                'impact': 'Medium',
                'description': f"High-risk customers have lower usage at {avg_usage_high_risk:.1f} vs. {avg_usage_overall:.1f} overall."
            })
    
    # Subscription Type Analysis
    if 'Subscription Type' in display_df.columns:
        # Map numeric values back to categories
        sub_mapping = {0: 'Basic', 1: 'Premium', 2: 'Standard'}
        display_df['Subscription Type Display'] = display_df['Subscription Type'].map(sub_mapping)
        
        # Calculate churn rate by subscription type
        churn_by_sub = display_df.groupby('Subscription Type Display')['Churn Probability'].mean()
        
        # Find subscription types with high churn probability
        high_churn_subs = churn_by_sub[churn_by_sub > churn_by_sub.mean() * 1.5]
        
        if not high_churn_subs.empty:
            insights.append({
                'category': 'Subscription Type',
                'impact': 'Medium',
                'description': f"Higher churn risk in: {', '.join(high_churn_subs.index)}"
            })
    
    # Display Insights
    if insights:
        for insight in insights:
            impact_color = {
                'High': 'red',
                'Medium': 'orange',
                'Low': 'green'
            }
            
            st.markdown(f"""
            <div style="background-color:{impact_color.get(insight['impact'], 'white')}; 
                        color:black; 
                        padding:10px; 
                        border-radius:5px; 
                        margin-bottom:10px;">
            🚨 **{insight['category']} Risk** ({insight['impact']} Impact)
            
            {insight['description']}
            </div>
            """, unsafe_allow_html=True)
        
        st.info("""
        💡 **Recommendations:**
        - Review and optimize customer support processes
        - Implement proactive customer engagement strategies
        - Consider personalized retention offers
        - Analyze and address factors contributing to customer dissatisfaction
        """)
    else:
        st.info("No significant churn risk insights found.")


def create_predictions_page():
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
                marker=dict(colors=['#EF5350', '#66BB6A', '#FFA726']) 
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
            
            # # Analyze categorical features if they exist
            # if 'Contract Length' in display_df.columns:
            #     # Map numeric values back to categories for display
            #     mapping = {0: 'Annual', 1: 'Monthly', 2: 'Quarterly'}
            #     display_df['Contract Length Display'] = display_df['Contract Length'].map(mapping)

            #     ### Debugging line:
            #     print(display_df['Risk Category'].unique())

            #     display_df['Risk Category'] = display_df['Risk Category'].astype(str).str.strip()
            #     # Calculate the percentage of high risk customers by contract type
            #     contract_risk = pd.crosstab(
            #         display_df['Contract Length Display'], 
            #         display_df['Risk Category'],
            #         normalize='index'
            #     ).reset_index()

            #     ### Debugging line:
            #     print(contract_risk.columns)
            #     print(contract_risk)

            #     # Melt the dataframe for easier plotting
            #     contract_risk_melted = pd.melt(
            #         contract_risk, 
            #         id_vars=['Contract Length Display'], 
            #         value_vars=['Low Risk', 'Medium Risk', 'High Risk'],
            #         var_name='Risk Category',
            #         value_name='Percentage'
            #     )
                
            #     # Multiply by 100 to get percentage
            #     contract_risk_melted['Percentage'] = contract_risk_melted['Percentage'] * 100
                
            #     fig = px.bar(
            #         contract_risk_melted,
            #         x='Contract Length Display',
            #         y='Percentage',
            #         color='Risk Category',
            #         title='Risk Category Distribution by Contract Length',
            #         template="plotly_dark",
            #         color_discrete_map={
            #             'Low Risk': '#66BB6A',
            #             'Medium Risk': '#FFA726',
            #             'High Risk': '#EF5350'
            #         }
            #     )
            #     st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Ensure we have high-risk customers
            high_risk_df = display_df[display_df['Risk Category'] == 'High Risk'].copy()
    
            if high_risk_df is not None:
                # Correlation Analysis
                st.subheader("🔗 Correlation Insights for High-Risk Customers")
                
                # Select numeric columns for correlation
                numeric_cols = ['Tenure', 'Usage Frequency', 'Support Calls', 
                                'Payment Delay', 'Total Spend', 'Last Interaction', 
                                'Churn Probability']
                numeric_cols = [col for col in numeric_cols if col in high_risk_df.columns]
                
                # Correlation Heatmap
                if len(numeric_cols) > 2:
                    correlation_matrix = high_risk_df[numeric_cols].corr()
                    
                    fig_corr = px.imshow(
                        correlation_matrix, 
                        text_auto=True, 
                        aspect="auto", 
                        title="Correlation Heatmap for High-Risk Customer Attributes",
                        color_continuous_scale='RdBu_r',
                        template="plotly_dark"
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                
                # Advanced Multivariate Analysis
                st.subheader("🎯 Multivariate Risk Profiling")
                
                # Create columns for comparative analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    # Scatter plot: Usage Frequency vs Churn Probability
                    if 'Usage Frequency' in high_risk_df.columns:
                        fig_scatter = px.scatter(
                            high_risk_df, 
                            x='Usage Frequency', 
                            y='Churn Probability',
                            color='Total Spend' if 'Total Spend' in high_risk_df.columns else None,
                            title='Usage Frequency Impact on Churn Risk',
                            labels={'Usage Frequency': 'Usage Frequency', 
                                    'Churn Probability': 'Churn Probability'},
                            template="plotly_dark"
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                
                with col2:
                    # Scatter plot: Support Calls vs Churn Probability
                    if 'Support Calls' in high_risk_df.columns:
                        fig_support = px.scatter(
                            high_risk_df, 
                            x='Support Calls', 
                            y='Churn Probability',
                            color='Tenure' if 'Tenure' in high_risk_df.columns else None,
                            title='Support Calls Impact on Churn Risk',
                            labels={'Support Calls': 'Number of Support Calls', 
                                    'Churn Probability': 'Churn Probability'},
                            template="plotly_dark"
                        )
                        st.plotly_chart(fig_support, use_container_width=True)
                
                st.subheader("🔍 Categorical Variable Risk Analysis")
                # Flexible column names check
                all_columns = high_risk_df.columns
                categorical_cols = ['Subscription Type', 'Contract Length']
                
                ### DEBUGGING
                # Print debug information
                #st.write("Available Columns:", list(all_columns))
                #st.write("Searching for Columns:", categorical_cols)
                
                for col in categorical_cols:
                    # Find the best matching column
                    matching_col = [c for c in all_columns if col.lower() in c.lower()]
                    
                    if not matching_col:
                        st.warning(f"No column found matching '{col}'")
                        continue
                    
                    matching_col = matching_col[0]
                    
                    # Flexible mapping with fallback
                    unique_values = high_risk_df[matching_col].unique()
                    st.write(f"Unique values in {matching_col}:", unique_values)
                    
                    # Create dynamic mapping based on actual data
                    mapping = {val: str(val) for val in unique_values}
                    
                    # Apply mapping
                    high_risk_df[f'{matching_col}_Display'] = high_risk_df[matching_col].map(mapping)
                    
                    # Aggregate churn probability by category
                    churn_by_category = high_risk_df.groupby(f'{matching_col}_Display')['Churn Probability'].agg(['mean', 'count']).reset_index()
                    churn_by_category['Percentage'] = churn_by_category['count'] / len(high_risk_df) * 100
                    
                    # Create two side-by-side charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Bar chart for mean churn probability
                        if not churn_by_category.empty:
                            fig_bar = px.bar(
                                churn_by_category,
                                x=f'{matching_col}_Display',
                                y='mean',
                                text=[f'{x:.2%}' for x in churn_by_category['mean']],
                                title=f'Mean Churn Probability by {matching_col}',
                                labels={
                                    f'{matching_col}_Display': matching_col,
                                    'mean': 'Mean Churn Probability'
                                },
                                template="plotly_dark"
                            )
                            fig_bar.update_traces(texttemplate='%{text}', textposition='outside')
                            st.plotly_chart(fig_bar, use_container_width=True)
                        else:
                            st.warning("No data for bar chart")
                    
                    with col2:
                        # Pie chart for high-risk customer distribution
                        if not churn_by_category.empty:
                            fig_pie = px.pie(
                                churn_by_category,
                                values='count',
                                names=f'{matching_col}_Display',
                                title=f'High-Risk Customer Distribution by {matching_col}',
                                hole=0.3,
                                template="plotly_dark"
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                        else:
                            st.warning("No data for pie chart")
                
                # Risk Mitigation Strategies
                st.subheader("🛡️ Risk Mitigation Strategies")
                
                # Create expandable sections for strategies
                strategies = [
                    {
                        'title': "Proactive Customer Engagement",
                        'description': "Develop personalized retention programs for high-risk customers based on their unique characteristics.",
                        'actions': [
                            "Create targeted communication plans",
                            "Offer personalized service upgrades",
                            "Provide special retention incentives"
                        ]
                    },
                    {
                        'title': "Support and Interaction Optimization",
                        'description': "Improve customer support and interaction quality to reduce churn risk.",
                        'actions': [
                            "Reduce support response times",
                            "Enhance customer support training",
                            "Implement proactive support strategies"
                        ]
                    },
                    {
                        'title': "Pricing and Value Proposition",
                        'description': "Reassess pricing strategies and value offerings for high-risk customer segments.",
                        'actions': [
                            "Create flexible pricing models",
                            "Develop segment-specific value packages",
                            "Conduct customer satisfaction surveys"
                        ]
                    }
                ]
                
                for strategy in strategies:
                    with st.expander(f"🎯 {strategy['title']}"):
                        st.markdown(f"**Description:** {strategy['description']}")
                        st.markdown("**Recommended Actions:**")
                        for action in strategy['actions']:
                            st.markdown(f"- {action}")        
            else:
                st.info("No high risk customers identified in the dataset.")
        
        # Add a new tab for Advanced Churn Analysis
        with st.expander("🔍 Advanced Churn Insights"):
            advanced_churn_analysis(display_df)
    else:
        st.info("Please upload data to see prediction results.")

def main():

    # Initialize session state if it doesn't exist
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'probabilities' not in st.session_state:
        st.session_state.probabilities = None
    if 'risk_categories' not in st.session_state:
        st.session_state.risk_categories = None
    
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
    
    model = load_model()
    # Validate and potentially modify the dataframe
    validated_data = validate_dataframe(data)
    if validated_data is not False:
        # Make predictions if data exists and is valid
        st.session_state.data = validated_data
        if st.session_state.data is not None and (st.session_state.predictions is None or uploaded_file is not None):
            predictions, probabilities = predict_churn(st.session_state.data, model)
            st.session_state.predictions = predictions
            st.session_state.probabilities = probabilities
            st.session_state.risk_categories = [categorize_risk(p) for p in probabilities]
            create_predictions_page()
    else:
        st.error("Please provide a dataset with the correct columns.")

if __name__ == "__main__":
    main()