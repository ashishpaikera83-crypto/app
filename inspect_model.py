import pickle
from pathlib import Path

path = Path('model.pkl')
print('exists', path.exists())
with open(path, 'rb') as f:
    obj = pickle.load(f)
print('type', type(obj))
print('class', getattr(obj, '__class__', None))
print('module', getattr(getattr(obj, '__class__', None), '__module__', None))
print('n_features_in_', getattr(obj, 'n_features_in_', None))
print('feature_names_in_', getattr(obj, 'feature_names_in_', None))
print('classes_', getattr(obj, 'classes_', None))
print('params', obj.get_params())
