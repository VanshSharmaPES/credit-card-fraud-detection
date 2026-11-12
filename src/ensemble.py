"""Average top validation-PR-AUC models after row-ID alignment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import matplotlib
matplotlib.use('Agg')
from evaluate import evaluate_model
from data import ROOT


def main():
 valfiles={p.stem.removesuffix('_val'):p for p in (ROOT/'preds').glob('*_val.csv') if not p.parent.name=='imbalance'}
 ranked=[]
 for name,path in valfiles.items():
  if name=='ensemble': continue
  d=pd.read_csv(path); ranked.append((name,float(average_precision_score(d.y_true,d.y_prob))))
 ranked.sort(key=lambda x:x[1],reverse=True)
 if len(ranked)<2: raise RuntimeError('Ensemble requires at least two models with validation predictions')
 selected=[n for n,_ in ranked[:3]]; print('Validation PR-AUC ranking:',ranked,'\nSelected:',selected)
 def average(split):
  frames=[pd.read_csv(ROOT/f'preds/{name}_{split}.csv').sort_values('row_id').reset_index(drop=True) for name in selected]
  base=frames[0][['row_id','y_true']].copy()
  for f in frames[1:]:
   if not np.array_equal(base.row_id,f.row_id) or not np.array_equal(base.y_true,f.y_true): raise ValueError('Predictions do not align by row_id/label')
  base['y_prob']=np.mean([f.y_prob.to_numpy() for f in frames],axis=0); return base
 v=average('val'); t=average('test')
 result=evaluate_model('ensemble',v.y_true,v.y_prob,t.y_true,t.y_prob,v.row_id,t.row_id,0,0,
  notes='Arithmetic mean of top three available models selected by validation PR-AUC; member timings excluded.',hardware='Average of saved model probabilities',output_dir=ROOT)
 (ROOT/'results/ensemble_members.json').write_text(json.dumps({'ranking':ranked,'selected':selected},indent=2)+'\n')
 print(result)
 # Cross-model validation permutation importance: one panel per available model.
 import matplotlib.pyplot as plt
 files=sorted((ROOT/'results').glob('perm_*.csv'))
 if files:
  fig,axes=plt.subplots(1,len(files),figsize=(5*len(files),6),sharey=False)
  if len(files)==1: axes=[axes]
  for ax,path in zip(axes,files):
   d=pd.read_csv(path).nlargest(10,'importance_mean').sort_values('importance_mean')
   ax.barh(d.feature,d.importance_mean,xerr=d.importance_std,color='#147D83')
   ax.set_title(path.stem.replace('perm_','')); ax.set_xlabel('Validation PR-AUC drop')
  fig.suptitle('Permutation importance by model (top 10 per model)'); fig.tight_layout()
  (ROOT/'figures').mkdir(exist_ok=True); fig.savefig(ROOT/'figures/perm_importance_all.png',dpi=180,bbox_inches='tight'); plt.close(fig)

if __name__=='__main__': main()
