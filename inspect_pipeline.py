import pickle

with open('model.pkl', 'rb') as handle:
    model = pickle.load(handle)

print(type(model))
print('pipeline feature_names_in_', getattr(model, 'feature_names_in_', None))
print('n_features_in_', getattr(model, 'n_features_in_', None))
print('steps', list(model.named_steps.keys()))
for name, step in model.named_steps.items():
    print('STEP', name, type(step))
    print('  feature_names_in', getattr(step, 'feature_names_in_', None))
    print('  n_features_in', getattr(step, 'n_features_in_', None))
