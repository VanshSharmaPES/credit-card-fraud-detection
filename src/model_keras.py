"""Keras MLP model using the common network and class-weight rule."""
from pathlib import Path
import sys,time,platform
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
sys.path.insert(0,str(Path(__file__).resolve().parent))
from data import ROOT,load_splits
from evaluate import evaluate_model,permutation_importance

def main():
 keras.utils.set_random_seed(42)
 Xtr,ytr,Xv,yv,Xte,yte,ridv,_,ridte,_=load_splits(); features=list(Xtr.columns); Xvdf=Xv.copy()
 Xtr=np.asarray(Xtr,dtype='float32'); Xv=np.asarray(Xv,dtype='float32'); Xte=np.asarray(Xte,dtype='float32')
 ytr=np.asarray(ytr,dtype='int32'); yv=np.asarray(yv,dtype='int32'); yte=np.asarray(yte,dtype='int32')
 pos=float((ytr==0).sum()/(ytr==1).sum())
 model=keras.Sequential([layers.Input((Xtr.shape[1],)),layers.Dense(64,activation='relu'),layers.Dropout(.2),layers.Dense(32,activation='relu'),layers.Dropout(.2),layers.Dense(1,activation='sigmoid')])
 model.compile(optimizer=keras.optimizers.Adam(.001),loss='binary_crossentropy',metrics=[keras.metrics.AUC(curve='PR',name='pr_auc')])
 stop=keras.callbacks.EarlyStopping(monitor='val_pr_auc',mode='max',patience=5,restore_best_weights=True)
 t=time.perf_counter(); model.fit(Xtr,ytr,validation_data=(Xv,yv),class_weight={0:1.0,1:pos},batch_size=2048,epochs=30,callbacks=[stop],verbose=2); train=time.perf_counter()-t
 pv=model.predict(Xv,batch_size=8192,verbose=0).ravel(); t=time.perf_counter(); pt=model.predict(Xte,batch_size=8192,verbose=0).ravel(); pred=time.perf_counter()-t
 (ROOT/'models').mkdir(exist_ok=True); model.save(ROOT/'models/keras.keras')
 res=evaluate_model('keras',yv,pv,yte,pt,ridv,ridte,train,pred,notes='Dense 64-32 ReLU; dropout 0.2; class_weight; Adam 0.001; early stopping on validation PR-AUC.',hardware=platform.machine()+' CPU',output_dir=ROOT)
 permutation_importance(lambda x:model.predict(np.asarray(x,dtype='float32'),batch_size=8192,verbose=0).ravel(),Xvdf,yv,3).to_csv(ROOT/'results/perm_keras.csv',index=False)
 print(res)
if __name__=='__main__': main()
