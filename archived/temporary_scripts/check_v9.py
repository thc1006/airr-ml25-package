import pandas as pd
import warnings
warnings.filterwarnings('ignore')

v9 = pd.read_csv('submission_ultimate.csv')
v9_preds = v9[v9['junction_aa'] == '-999.0'].copy()

print('V9 Predictions Summary:')
print('='*60)

for ds in sorted(v9_preds['dataset'].unique()):
    d = v9_preds[v9_preds['dataset'] == ds]['label_positive_probability']
    print(f'{ds:20s}: n={len(d):4d} mean={d.mean():.6f} std={d.std():.6f} min={d.min():.6f} max={d.max():.6f}')

print('\nOverall Statistics:')
all_preds = v9_preds['label_positive_probability']
print(f'Total predictions: {len(all_preds)}')
print(f'Mean: {all_preds.mean():.6f}')
print(f'Std: {all_preds.std():.6f}')
print(f'Min: {all_preds.min():.6f}')
print(f'Max: {all_preds.max():.6f}')
print(f'Predictions > 0.1: {(all_preds > 0.1).sum()} ({(all_preds > 0.1).sum()/len(all_preds)*100:.2f}%)')
print(f'Predictions > 0.5: {(all_preds > 0.5).sum()} ({(all_preds > 0.5).sum()/len(all_preds)*100:.2f}%)')
