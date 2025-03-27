import streamlit as st


# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Page Setup
dashboard = st.Page(
    page = "views/dashboard.py",
    title = "Dashboard",
    icon = "📊",
    default = True,
)

predictions = st.Page(
    page = "views/predictions.py",
    title = "Predictions",
    icon = "🔮",
)

# Navigation Setup
pg = st.navigation(pages = [dashboard, predictions])

# Sidebar Content
st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"]::before {
                content: "Customer Churn Prediction";
                margin-left: 25px;
                margin-top: 10px;
                font-size: 25px;
                position: relative;
                top: 2px;
            }
        </style>
        """,
        unsafe_allow_html=True,
)
st.sidebar.text("Customer Churn Prediction © 2025")

# Run Navigation
pg.run()