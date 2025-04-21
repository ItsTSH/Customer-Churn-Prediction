# 📉 Customer Churn Prediction Project

## 🚀 Overview
This machine learning project predicts customer churn using advanced data analysis and predictive modeling techniques. Deployed as an interactive web application using Streamlit, the project provides actionable insights into customer retention strategies.

## 📋 Table of Contents
- [🌟 Project Description](#-project-description)
- [💻 Installation](#-installation)
- [🚀 Deployment](#-deployment)
- [🗃️ Data](#️-data)
- [🤖 Model](#-model)
- [🔍 Usage](#-usage)
- [📊 Results](#-results)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

## 🌟 Project Description
Customer churn is a critical business challenge. This project offers:

| Feature | Description |
|---------|-------------|
| 🔮 Churn Prediction | Machine learning model to forecast customer attrition |
| 📊 Interactive Dashboard | Streamlit web app for real-time predictions |
| 🕵️ Insights Generation | Detailed analysis of churn factors |

### 🛠 Tech Stack
- Python
- Pandas
- Numpy
- Matplotlib
- Plotly
- Scikit-learn
- Streamlit
- TensorFlow

## 💻 Installation

### 🔧 Prerequisites
- Python 3.8+
- pip

### 📦 Setup
1. Clone the Repository:
```bash
git clone https://github.com/ItsTSH/Customer-Churn-Prediction
```

2. Create Virtual Environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```
3. Install Dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Deployment

### Streamlit App
```bash
# Run the Streamlit app
streamlit run app.py
```

## 🗃️ Data

### 📊 Dataset
- **Source:** [Kaggle](https://www.kaggle.com/)
- **Features:**
  - Customer demographics
  - Usage patterns
  - Service interactions
- **Target Variable:** Churn (Yes/No), Risk of Churn

## 🤖 Model

### 🧠 Machine Learning Algorithm(s)
- Random Forest Classifier (For Feature Importance Analysis)
- XGBoost Regressor (For Predictions)

### 📏 Evaluation Metrics
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

## 🔍 Usage

### 📱 Streamlit Application
1. Perform Data Analysis on the Dashboard Page
2. View Model Predictions in the Predictions Page
3. Receive churn prediction and insights

### 💻 Local Development
```bash
# Train the model
python ./model/Customer_Churn_Prediction.py

# Run predictions
python ./model/Customer_Churn_Prediction.py --input ./dataset/dataset.csv
```

## 📊 Project Results
- **Evaluation Metrics (XGBoost Regression Model):**
1. Mean Squared Error (MSE): 0.0450
2. Root Mean Squared Error (RMSE): 0.2120
3. Mean Absolute Error (MAE): 0.1292
4. R² Score: 0.8168 

## 🤝 Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch 
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your changes
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. Push to the branch
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a Pull Request

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.

## 📞 Contact
- 📧 **Email:** [tejindersingh0784@gmail.com]
- 🔗 **GitHub:** [https://github.com/ItsTSH]
- 📧 **Email:** [sashankskmishra@gmail.com]
- 🔗 **GitHub:** [https://github.com/sskm664]
- 📧 **Email:** [suyashart30@gmail.com]
- 🔗 **GitHub:** [https://github.com/SuyashArt]
---

⭐ If you find this project helpful, please consider starring the repository! 🌟
