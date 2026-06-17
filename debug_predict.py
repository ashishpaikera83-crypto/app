import pickle
import pandas as pd

with open('model.pkl', 'rb') as handle:
    model = pickle.load(handle)

sample = {'Area': 'North', 'Item': 'Rice', 'Element': 'Production', 'Year': 2020, 'Unit': 1}
df = pd.DataFrame([sample])
print(df)
try:
    prediction = model.predict(df)
    print('prediction', prediction)
except Exception as exc:
    import traceback
    traceback.print_exc()
