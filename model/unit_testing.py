import unittest 
import numpy as np
from Customer_Churn_Prediction import get_risk_category, evaluate_regression_model

class TestFunctions(unittest.TestCase): 

    def test_get_risk_category_single(self):
        self.assertEqual(get_risk_category(0.1), 'Low')
        self.assertEqual(get_risk_category(0.5), 'Medium')
        self.assertEqual(get_risk_category(0.9), 'High')

    def test_get_risk_category_list(self):
        test_probs = [0.1, 0.5, 0.9]
        expected = ['Low', 'Medium', 'High']
        result = get_risk_category(test_probs)
        self.assertEqual(result, expected)

    def test_get_risk_category_numpy_array(self):
        test_probs = np.array([0.2, 0.4, 0.8])
        expected = ['Low', 'Medium', 'High']
        result = get_risk_category(test_probs)
        self.assertEqual(result, expected)

    def test_evaluate_regression_model_output_keys(self):
        y_true = np.array([3, 5, 2.5])
        y_pred = np.array([2.5, 5, 3])
        result = evaluate_regression_model(y_true, y_pred)
        self.assertIn('mse', result)
        self.assertIn('rmse', result)
        self.assertIn('mae', result)
        self.assertIn('r2', result)

    def test_evaluate_regression_model_values(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])
        result = evaluate_regression_model(y_true, y_pred)
        self.assertEqual(result['mse'], 0.0)
        self.assertEqual(result['rmse'], 0.0)
        self.assertEqual(result['mae'], 0.0)
        self.assertEqual(result['r2'], 1.0)

if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=2, exit=False)