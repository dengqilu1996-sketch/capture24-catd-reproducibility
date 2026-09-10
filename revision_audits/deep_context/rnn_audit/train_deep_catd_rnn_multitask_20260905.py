"""Train a compact 10-class multi-head GRU and evaluate CATD on its posteriors.

This is a supplementary, current-task experiment.  Fine, coarse, and MET-intensity
heads share a bidirectional GRU trunk and are trained on P001--P080.  P081--P100
is used only for CATD operating-point selection; P101--P151 is locked for testing.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
torch.set_num_threads(min(8, os.cpu_count() or 8))

ROOT=Path(r"D:/capture24_project_current")
DATA=ROOT/"data/c_drive_large_capture24_migration_20260701/capture24_deep_data"
FIELDS=ROOT/"results/bcm/capture24_derived_label_fields_20260623_152248.npz"
OUT=ROOT/"results/deep_rnn_catd_20260905"
OUT.mkdir(parents=True,exist_ok=True)

FINE=np.array(["bicycling","household-chores","manual-work","mixed-activity","sitting","sleep","sports","standing","vehicle","walking"])
COARSE=np.array(["light","moderate-vigorous","sedentary","sleep"])
INTENS=np.array(["light","mvpa","sedentary","sleep"])

class WindowDataset(Dataset):
    def __init__(self,X,idx,yf,yc,yi): self.X=X; self.idx=idx; self.yf=yf; self.yc=yc; self.yi=yi
    def __len__(self): return len(self.idx)
    def __getitem__(self,j):
        i=int(self.idx[j]); x=torch.from_numpy(np.asarray(self.X[i],dtype=np.float32))
        return x, torch.tensor(self.yf[j],dtype=torch.long), torch.tensor(self.yc[j],dtype=torch.long), torch.tensor(self.yi[j],dtype=torch.long)

class MultiHeadRNN(nn.Module):
    def __init__(self,nf,nc,ni):
        super().__init__()
        self.trunk=nn.GRU(input_size=3,hidden_size=64,num_layers=2,batch_first=True,
                          dropout=.2,bidirectional=True)
        self.shared=nn.Sequential(nn.Linear(128,128),nn.ReLU(inplace=True),nn.Dropout(.2))
        self.fine=nn.Linear(128,nf); self.coarse=nn.Linear(128,nc); self.intensity=nn.Linear(128,ni)
    def forward(self,x):
        # Downsample 1000 Hz-window samples to 100 recurrent steps for a
        # tractable current-task audit while preserving the full window span.
        x=x[:,::10,:]
        h,_=self.trunk(x); h=self.shared(h[:,-1,:])
        return self.fine(h),self.coarse(h),self.intensity(h)

def macro(y,p): return float(metrics.f1_score(y,p,average='macro',zero_division=0))
def weight(y,n):
    c=np.bincount(y,minlength=n).astype(float); w=1/np.sqrt(np.maximum(c,1)); return torch.tensor(w/w.mean(),dtype=torch.float32)

def evaluate(model,loader,device,losses=None,save_logits=False):
    model.eval(); ys=[[],[],[]]; ps=[[],[],[]]; ls=[]; all_logits=[[],[],[]]
    with torch.no_grad():
        for x,yf,yc,yi in loader:
            x=x.to(device,non_blocking=True)
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=device.type=='cuda'):
                lf,lc,li=model(x)
                if losses is not None: ls.append(float(losses[0](lf,yf.to(device))+0.3*losses[1](lc,yc.to(device))+0.3*losses[2](li,yi.to(device))))
            for j,(logit,y) in enumerate([(lf,yf),(lc,yc),(li,yi)]):
                ys[j].append(y.numpy()); ps[j].append(logit.argmax(1).cpu().numpy())
                if save_logits: all_logits[j].append(logit.float().cpu().numpy())
    yy=[np.concatenate(a) for a in ys]; pp=[np.concatenate(a) for a in ps]
    out={'fine_macro_f1':macro(yy[0],pp[0]),'coarse_macro_f1':macro(yy[1],pp[1]),'intensity_macro_f1':macro(yy[2],pp[2]),'loss':float(np.mean(ls)) if ls else None}
    if save_logits: out['logits']=[np.concatenate(a).astype('float32') for a in all_logits]
    return out,yy,pp

def transition(y,p,classes,smooth=1e-3):
    m=np.full((len(classes),len(classes)),smooth,float); ix={c:i for i,c in enumerate(classes)}
    for pid in np.unique(p):
        s=y[p==pid]
        for a,b in zip(s[:-1],s[1:]): m[ix[a],ix[b]]+=1
    return m/m.sum(1,keepdims=True)

def viterbi(proba,p,classes,tr,gamma):
    lt=gamma*np.log(np.maximum(tr,1e-12)); out=np.empty(len(p),object); ix={c:i for i,c in enumerate(classes)}
    for pid in np.unique(p):
        ind=np.where(p==pid)[0]; e=np.log(np.maximum(proba[ind],1e-12)); n,k=e.shape; dp=np.empty((n,k)); bk=np.zeros((n,k),int); dp[0]=e[0]
        for t in range(1,n):
            sc=dp[t-1][:,None]+lt; bk[t]=sc.argmax(0); dp[t]=e[t]+sc[bk[t],np.arange(k)]
        path=np.zeros(n,int); path[-1]=dp[-1].argmax()
        for t in range(n-2,-1,-1): path[t]=bk[t+1,path[t+1]]
        out[ind]=classes[path]
    return out

def main():
    z=np.load(FIELDS,allow_pickle=True); valid=z['valid_annotation_mask'];
    # Deep X/Y are exactly the valid rows of the derived-label archive.
    p=z['participant'][valid]; yf0=z['y_willetts_specific2018'][valid]; yc0=z['y_walmsley2020'][valid]; yi0=z['y_met_intensity4'][valid]
    X=np.load(DATA/'X.npy',mmap_mode='r')
    if len(X)!=len(p): raise ValueError('Deep X and valid label archive are not aligned.')
    fidx={c:i for i,c in enumerate(FINE)}; cidx={c:i for i,c in enumerate(COARSE)}; iidx={c:i for i,c in enumerate(INTENS)}
    yf=np.array([fidx[x] for x in yf0],dtype=np.int64); yc=np.array([cidx[x] for x in yc0],dtype=np.int64); yi=np.array([iidx[x] for x in yi0],dtype=np.int64)
    train=np.isin(p,[f'P{i:03d}' for i in range(1,81)]); val=np.isin(p,[f'P{i:03d}' for i in range(81,101)]); test=np.isin(p,[f'P{i:03d}' for i in range(101,152)])
    # Keep chronological order within each participant; no random shuffling for evaluation.
    ti=np.flatnonzero(train); vi=np.flatnonzero(val); si=np.flatnonzero(test)
    tr=WindowDataset(X,ti,yf[ti],yc[ti],yi[ti]); va=WindowDataset(X,vi,yf[vi],yc[vi],yi[vi]); te=WindowDataset(X,si,yf[si],yc[si],yi[si])
    # Windows worker spawning cannot safely pickle the large read-only memmap;
    # use the main process so the experiment remains deterministic and robust.
    dltr=DataLoader(tr,batch_size=1024,shuffle=True,num_workers=0,pin_memory=True)
    dlva=DataLoader(va,batch_size=2048,shuffle=False,num_workers=0,pin_memory=True)
    dlte=DataLoader(te,batch_size=2048,shuffle=False,num_workers=0,pin_memory=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=MultiHeadRNN(len(FINE),len(COARSE),len(INTENS)).to(device)
    lf=nn.CrossEntropyLoss(weight=weight(yf[ti],len(FINE)).to(device)); lc=nn.CrossEntropyLoss(weight=weight(yc[ti],len(COARSE)).to(device)); li=nn.CrossEntropyLoss(weight=weight(yi[ti],len(INTENS)).to(device)); losses=(lf,lc,li)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=8); scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    best=-1; history=[]
    # Mandatory small dry-run before full training.
    xb,yfb,ycb,yib=next(iter(DataLoader(tr,batch_size=8,shuffle=False,num_workers=0))); xb=xb.to(device)
    with torch.no_grad():
        q=model(xb); assert q[0].shape==(8,len(FINE)) and q[1].shape==(8,len(COARSE)) and q[2].shape==(8,len(INTENS))
    for ep in range(1,9):
        model.train(); total=0.; nb=0
        for x,yb,cb,ib in dltr:
            x=x.to(device,non_blocking=True); yb=yb.to(device,non_blocking=True); cb=cb.to(device,non_blocking=True); ib=ib.to(device,non_blocking=True); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=device.type=='cuda'):
                a,b,c=model(x); loss=lf(a,yb)+.3*lc(b,cb)+.3*li(c,ib)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); total+=float(loss); nb+=1
        sched.step(); valres,_,_=evaluate(model,dlva,device,losses); history.append({'epoch':ep,'train_loss':total/max(nb,1),**{k:v for k,v in valres.items() if k!='logits'}}); print(history[-1],flush=True)
        if valres['fine_macro_f1']>best:
            best=valres['fine_macro_f1']; torch.save({'model':model.state_dict(),'epoch':ep,'val':valres},OUT/'best.pt')
    ck=torch.load(OUT/'best.pt',map_location=device,weights_only=False); model.load_state_dict(ck['model']); valres,vy,vp=evaluate(model,dlva,device,losses,save_logits=True); testres,ty,tp=evaluate(model,dlte,device,losses,save_logits=True)
    # Save posterior arrays in participant/time order for CATD decoding.
    np.savez_compressed(OUT/'posteriors.npz',participant=p[vi],y_fine=yf0[vi],fine_logits=valres['logits'][0],coarse_logits=valres['logits'][1],intensity_logits=valres['logits'][2],val_index=vi)
    np.savez_compressed(OUT/'posteriors_test.npz',participant=p[si],y_fine=yf0[si],fine_logits=testres['logits'][0],coarse_logits=testres['logits'][1],intensity_logits=testres['logits'][2],test_index=si)
    # Keep posterior logits in the NPZ audit files, but omit ndarray payloads
    # from the human-readable JSON training report.
    def clean_result(r): return {k:v for k,v in r.items() if k != 'logits'}
    json.dump({'history':history,'best_epoch':int(ck['epoch']),'val':clean_result(valres),'test':clean_result(testres),'classes':{'fine':FINE.tolist(),'coarse':COARSE.tolist(),'intensity':INTENS.tolist()}},open(OUT/'training_report.json','w',encoding='utf-8'),indent=2)
    # CATD: map auxiliary posterior heads to fine labels using P001-P080 majority mappings.
    def majority(src,tgt):
        m=np.zeros((len(FINE),len(tgt)),float)
        for a,b in zip(yf0[ti],src[ti]): m[fidx[a],{x:i for i,x in enumerate(tgt)}[b]]+=1
        return m/np.maximum(m.sum(1,keepdims=True),1)
    mf2c=majority(yc0,COARSE); mf2i=majority(yi0,INTENS); trm=transition(yf0[ti],p[ti],FINE)
    pv=np.exp(valres['logits'][0]-valres['logits'][0].max(1,keepdims=True)); pv/=pv.sum(1,keepdims=True); pc=np.exp(valres['logits'][1]-valres['logits'][1].max(1,keepdims=True)); pc/=pc.sum(1,keepdims=True); pi=np.exp(valres['logits'][2]-valres['logits'][2].max(1,keepdims=True)); pi/=pi.sum(1,keepdims=True)
    ps=np.exp(testres['logits'][0]-testres['logits'][0].max(1,keepdims=True)); ps/=ps.sum(1,keepdims=True); pcs=np.exp(testres['logits'][1]-testres['logits'][1].max(1,keepdims=True)); pcs/=pcs.sum(1,keepdims=True); pis=np.exp(testres['logits'][2]-testres['logits'][2].max(1,keepdims=True)); pis/=pis.sum(1,keepdims=True)
    csv=[]
    for a in [0,.25,.5,1]:
      for b in [0,.25,.5,1]:
       em=pv*np.power(np.maximum(pc@mf2c.T,1e-12),a)*np.power(np.maximum(pi@mf2i.T,1e-12),b)
       for g in [0,.5,1,2,3,4,5,6]:
        pred=viterbi(em,p[vi],FINE,trm,g); csv.append({'alpha':a,'beta':b,'gamma':g,'macro_f1':macro(yf0[vi],pred),'balanced_accuracy':metrics.balanced_accuracy_score(yf0[vi],pred)})
    gd=pd.DataFrame(csv); bestm=gd.macro_f1.max(); el=gd[gd.macro_f1>=.95*bestm]; sel=el.sort_values(['macro_f1','balanced_accuracy'],ascending=False).iloc[0]
    # temporal-only uses alpha=beta=0 and its best gamma.
    to=gd[(gd.alpha==0)&(gd.beta==0)].sort_values(['macro_f1','balanced_accuracy'],ascending=False).iloc[0]
    ems=ps*np.power(np.maximum(pcs@mf2c.T,1e-12),float(sel.alpha))*np.power(np.maximum(pis@mf2i.T,1e-12),float(sel.beta)); predc=viterbi(ems,p[si],FINE,trm,float(sel.gamma)); predt=viterbi(ps,p[si],FINE,trm,float(to.gamma)); predh=viterbi(ps,p[si],FINE,trm,1)
    rows=[]
    for name,pred in [('rnn',FINE[ np.argmax(testres['logits'][0],axis=1)]),('rnn_hmm',predh),('rnn_temporal_only',predt),('rnn_catd',predc)]: rows.append({'method':name,'alpha':float(sel.alpha) if name=='rnn_catd' else (0 if name=='rnn_temporal_only' else np.nan),'beta':float(sel.beta) if name=='rnn_catd' else (0 if name=='rnn_temporal_only' else np.nan),'gamma':float(sel.gamma) if name=='rnn_catd' else (float(to.gamma) if name=='rnn_temporal_only' else (1 if name=='rnn_hmm' else np.nan)),'macro_f1':macro(yf0[si],pred),'balanced_accuracy':metrics.balanced_accuracy_score(yf0[si],pred)})
    gd.to_csv(OUT/'validation_candidates.csv',index=False); pd.DataFrame(rows).to_csv(OUT/'locked_test_metrics.csv',index=False); json.dump({'selected':sel.to_dict(),'temporal_only':to.to_dict(),'locked_rows':rows},open(OUT/'catd_report.json','w',encoding='utf-8'),indent=2)
    print('CATD',sel.to_dict()); print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__': main()
