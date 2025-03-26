import streamlit as st

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
st.sidebar.text("Customer Churn Prediction ©️ 2025")

# Run Navigation
pg.run()