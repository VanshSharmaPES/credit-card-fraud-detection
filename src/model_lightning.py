"""Plain PyTorch Lightning MLP fallback (Flash is archived per project plan)."""
from pathlib import Path
import sys,time,platform
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping,ModelCheckpoint
from torchmetrics.classification import BinaryAveragePrecision
sys.path.insert(0,str(Path(__file__).resolve().parent))
from data import ROOT,load_splits
from evaluate import evaluate_model,permutation_importance

class FraudModule(L.LightningModule):
 def __init__(self,n_features,pos_weight):
  super().__init__(); self.net=nn.Sequential(nn.Linear(n_features,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,32),nn.ReLU(),nn.Dropout(.2),nn.Linear(32,1))
  self.register_buffer('pos_weight',torch.tensor([pos_weight],dtype=torch.float32)); self.loss_fn=nn.BCEWithLogitsLoss(pos_weight=self.pos_weight); self.val_ap=BinaryAveragePrecision()
 def forward(self,x): return self.net(x).squeeze(1)
 def training_step(self,batch,batch_idx):
  x,y=batch; loss=self.loss_fn(self(x),y.float()); self.log('train_loss',loss,on_epoch=True); return loss
 def validation_step(self,batch,batch_idx):
  x,y=batch; logits=self(x); self.val_ap.update(torch.sigmoid(logits),y); self.log('val_loss',self.loss_fn(logits,y.float()),on_epoch=True,on_step=False); self.log('val_pr_auc',self.val_ap,on_epoch=True,on_step=False,prog_bar=True)
 def configure_optimizers(self): return torch.optim.Adam(self.parameters(),lr=.001)

def main():
 L.seed_everything(42,workers=True); Xtr,ytr,Xv,yv,Xte,yte,ridv,_,ridte,_=load_splits(); names=list(Xtr.columns)
 a=lambda x:torch.tensor(np.asarray(x,dtype=np.float32))
 y=lambda x:torch.tensor(np.asarray(x,dtype=np.int64))
 pos=float((ytr==0).sum()/(ytr==1).sum()); tr=DataLoader(TensorDataset(a(Xtr),y(ytr)),batch_size=2048,shuffle=True); va=DataLoader(TensorDataset(a(Xv),y(yv)),batch_size=2048)
 ROOT.joinpath('models').mkdir(exist_ok=True); cb=ModelCheckpoint(dirpath=ROOT/'models/lightning_checkpoints',monitor='val_pr_auc',mode='max',save_top_k=1,filename='best-{epoch:02d}-{val_pr_auc:.4f}')
 trainer=L.Trainer(max_epochs=30,accelerator='cpu',devices=1,callbacks=[cb,EarlyStopping(monitor='val_pr_auc',mode='max',patience=5)],logger=False,enable_progress_bar=True,enable_model_summary=False)
 model=FraudModule(Xtr.shape[1],pos); t=time.perf_counter(); trainer.fit(model,tr,va); train=time.perf_counter()-t
 best=FraudModule.load_from_checkpoint(cb.best_model_path,n_features=Xtr.shape[1],pos_weight=pos,map_location='cpu'); best.eval()
 def predict(X):
  loader=DataLoader(a(X),batch_size=8192); out=[]
  with torch.no_grad():
   for x in loader: out.append(torch.sigmoid(best(x)).numpy())
  return np.concatenate(out)
 pv=predict(Xv); t=time.perf_counter(); pt=predict(Xte); pred=time.perf_counter()-t
 import shutil; shutil.copy2(cb.best_model_path,ROOT/'models/lightning.ckpt')
 res=evaluate_model('lightning',yv,pv,yte,pt,ridv,ridte,train,pred,notes='Plain PyTorch Lightning; shared dense 64-32 network; weighted BCEWithLogitsLoss; early stopping on validation PR-AUC.',hardware=platform.machine()+' CPU',output_dir=ROOT)
 permutation_importance(lambda x:predict(x),pd.DataFrame(Xv,columns=names),yv,3).to_csv(ROOT/'results/perm_lightning.csv',index=False)
 print(res)
if __name__=='__main__': main()
