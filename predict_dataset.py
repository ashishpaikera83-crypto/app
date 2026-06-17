import pickle
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load model
with open('model.pkl', 'rb') as handle:
    model = pickle.load(handle)

# Load dataset
df = pd.read_csv('jk.csv')

# Prepare features for prediction
features = ['Area', 'Item', 'Element', 'Year', 'Unit']
X = df[features]
y_actual = df['Value']

# Remove rows with NaN values
valid_mask = ~(X.isna().any(axis=1) | y_actual.isna())
X = X[valid_mask]
y_actual = y_actual[valid_mask]

# Make predictions
y_pred = model.predict(X)

# Calculate metrics
mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))

# Create results dataframe
results = df[valid_mask].copy()
results['Predicted_Value'] = y_pred
results['Error'] = abs(y_actual.values - y_pred)
results['Error_Percent'] = (abs(y_actual.values - y_pred) / y_actual.values * 100).round(2)

print('='*100)
print('MODEL PREDICTIONS ON FULL DATASET')
print('='*100)
print(f'Total Records: {len(results)}')
print(f'Mean Absolute Error (MAE): {mae:,.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse:,.2f}')
print('\n' + '='*100)
print('SAMPLE PREDICTIONS (First 15 rows):')
print('='*100)
sample_cols = ['Area', 'Item', 'Year', 'Value', 'Predicted_Value', 'Error', 'Error_Percent']
print(results[sample_cols].head(15).to_string(index=False))
print('\n' + '='*100)
print('SUMMARY STATISTICS:')
print('='*100)
print(f'Minimum Error: {results["Error"].min():,.2f}')
print(f'Maximum Error: {results["Error"].max():,.2f}')
print(f'Average Error: {results["Error"].mean():,.2f}')
print(f'Median Error: {results["Error"].median():,.2f}')
print(f'Average Error %: {results["Error_Percent"].mean():.2f}%')

# Save detailed results to CSV
results.to_csv('predictions_output.csv', index=False)
print(f'\nDetailed results saved to: predictions_output.csv')
