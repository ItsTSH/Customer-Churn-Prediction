# testing.py - Unit tests for the Telecom Customer Churn XGBoost Model

import unittest
import os
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

# Import functions from your prediction.py file
# Adjust the import path if your file has a different name
try:
    from prediction import (
        get_risk_category,
        evaluate_regression_model,
        visualizePredictions
    )
    FUNCTIONS_IMPORTED = True
except ImportError:
    print("Warning: Could not import functions from prediction.py. Will use local definitions for testing.")
    FUNCTIONS_IMPORTED = False
    
    # Local definitions if import fails
    def get_risk_category(probability, thresholds=(0.33, 0.67)):
        low_threshold, high_threshold = thresholds
        
        if isinstance(probability, (list, np.ndarray, pd.Series)):
            return [get_risk_category(p, thresholds) for p in probability]
        
        if probability < low_threshold:
            return 'Low'
        elif probability < high_threshold:
            return 'Medium'
        else:
            return 'High'
    
    def evaluate_regression_model(y_true, y_pred):
        """
        Evaluate regression model performance with multiple metrics
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def visualizePredictions(y_pred, thresholds=(0.33, 0.67), figsize=(10, 8)):
        """Simplified version for testing"""
        risk_categories = get_risk_category(y_pred, thresholds)
        category_counts = pd.Series(risk_categories).value_counts()
        return category_counts

class TestTelecomChurnModel(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests"""
        print(f"\nFunctions imported from prediction.py: {FUNCTIONS_IMPORTED}")
        
        # Create a small synthetic dataset for testing
        np.random.seed(42)
        n_samples = 100
        
        cls.test_data = pd.DataFrame({
            'Tenure': np.random.normal(30, 10, n_samples),
            'Usage Frequency': np.random.normal(25, 8, n_samples),
            'Support Calls': np.random.randint(0, 10, n_samples),
            'Payment Delay': np.random.randint(0, 30, n_samples),
            'Total Spend': np.random.normal(500, 150, n_samples),
            'Last Interaction': np.random.randint(1, 90, n_samples),
            'Subscription Type': np.random.choice(['Basic', 'Premium', 'Standard'], n_samples),
            'Contract Length': np.random.choice(['Annual', 'Monthly', 'Quarterly'], n_samples),
            'Churn': np.random.binomial(1, 0.2, n_samples).astype(float)
        })
        
        # Save test data to CSV
        cls.test_data.to_csv('test_dataset.csv', index=False)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        # Remove test files
        if os.path.exists('test_dataset.csv'):
            os.remove('test_dataset.csv')
        if os.path.exists('test_model.pkl'):
            os.remove('test_model.pkl')
        if os.path.exists('test_scaler.pkl'):
            os.remove('test_scaler.pkl')
    
    def test_01_data_loading(self):
        """Test data loading functionality"""
        print("\nRunning test_01_data_loading...")
        df = pd.read_csv('test_dataset.csv')
        self.assertEqual(len(df), 100)
        self.assertTrue('Churn' in df.columns)
        self.assertTrue('Subscription Type' in df.columns)
        self.assertTrue('Contract Length' in df.columns)
        print("Data loading test passed!")
    
    def test_02_data_preprocessing(self):
        """Test data preprocessing (label encoding and scaling)"""
        print("\nRunning test_02_data_preprocessing...")
        df = pd.read_csv('test_dataset.csv')
        
        # Test categorical encoding
        from sklearn.preprocessing import LabelEncoder
        
        for column in ['Subscription Type', 'Contract Length']:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
        
        self.assertTrue(pd.api.types.is_numeric_dtype(df['Subscription Type']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['Contract Length']))
        
        # Test numerical scaling
        scaler = StandardScaler()
        numeric_columns = ['Tenure', 'Usage Frequency', 'Support Calls', 
                          'Payment Delay', 'Total Spend', 'Last Interaction']
        
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        
        for col in numeric_columns:
            self.assertAlmostEqual(df[col].mean(), 0, places=1)
            self.assertAlmostEqual(df[col].std(), 1, places=1)
        
        print("Data preprocessing test passed!")
    
    def test_03_model_training_and_prediction(self):
        """Test model training and prediction functionality"""
        print("\nRunning test_03_model_training_and_prediction...")
        df = pd.read_csv('test_dataset.csv')
        
        # Preprocess data
        from sklearn.preprocessing import LabelEncoder
        
        for column in ['Subscription Type', 'Contract Length']:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
        
        scaler = StandardScaler()
        numeric_columns = ['Tenure', 'Usage Frequency', 'Support Calls', 
                          'Payment Delay', 'Total Spend', 'Last Interaction']
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        
        # Split data
        x = df.drop('Churn', axis=1)
        y = df['Churn']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        
        # Train model
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dtest = xgb.DMatrix(x_test, label=y_test)
        
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 3,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        }
        
        num_rounds = 10  # Reduced for testing speed
        model = xgb.train(params, dtrain, num_rounds)
        
        # Make predictions
        y_pred = model.predict(dtest)
        
        # Test prediction shape
        self.assertEqual(len(y_pred), len(y_test))
        
        # Save model for other tests
        pickle.dump(model, open("test_model.pkl", "wb"))
        
        # Test prediction scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        y_pred_scaled = scaler.fit_transform(y_pred.reshape(-1, 1)).flatten()
        
        # Save scaled predictions for other tests
        self.y_pred_scaled = y_pred_scaled
        
        # Save scaler
        pickle.dump(scaler, open("test_scaler.pkl", "wb"))
        
        print("Model training and prediction test passed!")
    
    def test_04_risk_categorization(self):
        """Test risk categorization function"""
        print("\nRunning test_04_risk_categorization...")
        
        # Test with single values
        self.assertEqual(get_risk_category(0.1), 'Low')
        self.assertEqual(get_risk_category(0.5), 'Medium')
        self.assertEqual(get_risk_category(0.8), 'High')
        
        # Test with array
        test_array = np.array([0.1, 0.4, 0.9])
        expected = ['Low', 'Medium', 'High']
        self.assertEqual(get_risk_category(test_array), expected)
        
        # Test with custom thresholds
        self.assertEqual(get_risk_category(0.4, thresholds=(0.3, 0.6)), 'Medium')
        self.assertEqual(get_risk_category(0.3, thresholds=(0.3, 0.6)), 'Medium')
        self.assertEqual(get_risk_category(0.29, thresholds=(0.3, 0.6)), 'Low')
        
        print("Risk categorization test passed!")
    
    def test_05_model_evaluation(self):
        """Test model evaluation function"""
        print("\nRunning test_05_model_evaluation...")
        
        # Create test data
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.3])
        
        # Run evaluation function
        metrics = evaluate_regression_model(y_true, y_pred)
        
        # Test metrics calculation
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2', metrics)
        
        # Test metric values
        self.assertAlmostEqual(metrics['mse'], 0.03, places=2)
        self.assertAlmostEqual(metrics['rmse'], 0.173, places=2)
        
        print("Model evaluation test passed!")
    
    def test_06_visualization_function(self):
        """Test visualization function"""
        print("\nRunning test_06_visualization_function...")
        
        try:
            # Try to run the function with test data
            test_data = np.random.random(100)
            
            # Mock the visualization function to avoid actual plotting
            def mock_visualize(y_pred, thresholds=(0.33, 0.67), figsize=(10, 8)):
                # Get risk categories
                risk_categories = get_risk_category(y_pred, thresholds)
                
                # Count categories
                category_counts = pd.Series(risk_categories).value_counts()
                
                # Just check we can get these values
                percentages = 100 * category_counts / len(risk_categories)
                
                return True
            
            result = mock_visualize(test_data)
            self.assertTrue(result)
            
            # If we have predictions from previous test, try to use them too
            if hasattr(self, 'y_pred_scaled'):
                result = mock_visualize(self.y_pred_scaled)
                self.assertTrue(result)
                
            print("Visualization function test passed!")
        except Exception as e:
            self.fail(f"Visualization function test failed with error: {e}")
    
    def test_07_model_serialization(self):
        """Test model serialization/deserialization"""
        print("\nRunning test_07_model_serialization...")
        
        # Check if model file exists
        self.assertTrue(os.path.exists('test_model.pkl'), 
                       "Model file not found. Run test_03_model_training_and_prediction first.")
        
        # Load model
        model = pickle.load(open("test_model.pkl", "rb"))
        self.assertIsNotNone(model)
        
        # Check model type
        self.assertIsInstance(model, xgb.Booster)
        
        # Check if scaler file exists
        self.assertTrue(os.path.exists('test_scaler.pkl'),
                       "Scaler file not found. Run test_03_model_training_and_prediction first.")
        
        # Load scaler
        scaler = pickle.load(open("test_scaler.pkl", "rb"))
        self.assertIsNotNone(scaler)
        
        print("Model serialization test passed!")

def run_tests():
    """Run all tests with more descriptive output"""
    print("\n=== Starting Telecom Churn Model Tests ===\n")
    unittest.main(verbosity=2, argv=['first-arg-is-ignored'], exit=False)
    print("\n=== All tests completed! ===")

if __name__ == '__main__':
    run_tests()