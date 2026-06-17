import pandas as pd
import pickle

# Load dataset
df = pd.read_csv('jk.csv')

# Filter for Afghanistan Asses
afghanistan_asses = df[(df['Area'] == 'Afghanistan') & (df['Item'] == 'Asses')]

print('Afghanistan Asses Data:')
print(afghanistan_asses[['Year', 'Value']].to_string(index=False))
print(f'\nTotal records: {len(afghanistan_asses)}')
print(f'Year range: {afghanistan_asses["Year"].min()} - {afghanistan_asses["Year"].max()}')

# Load model and test prediction
with open('model.pkl', 'rb') as handle:
    model = pickle.load(handle)

# Test prediction for 2020
test_data = pd.DataFrame([{
    'Area': 'Afghanistan',
    'Item': 'Asses',
    'Element': 'Stocks',
    'Year': 2020,
    'Unit': 'Head'
}])

prediction = model.predict(test_data)[0]
print(f'\nModel prediction for 2020: {prediction:,.2f}')

# Compare with dataset patterns
print(f'\nDataset statistics for Afghanistan Asses:')
print(f'Min value: {afghanistan_asses["Value"].min():,.0f}')
print(f'Max value: {afghanistan_asses["Value"].max():,.0f}')
print(f'Mean value: {afghanistan_asses["Value"].mean():,.0f}')
print(f'Last recorded year ({afghanistan_asses["Year"].max()}): {afghanistan_asses[afghanistan_asses["Year"] == afghanistan_asses["Year"].max()]["Value"].values[0]:,.0f}')
