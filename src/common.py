from pathlib import Path
import sys, os, json, csv, hashlib, time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'runtime_tmp'/'mpl'))
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
import torch
torch.set_num_threads(2)
DATA=ROOT/'inputs'

def dump(name,obj):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item()),encoding='utf-8')

def table(name,rows):
    rows=list(rows)
    if not rows:return
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def matrix(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.reader(f);cols=next(reader)[1:];names=[];values=[]
        for row in reader:names.append(row[0]);values.append([float(v) for v in row[1:]])
    return names,cols,np.asarray(values,dtype=np.float64)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def load_data():
    sp,sa,A=matrix(DATA/'species_abundance.csv')
    fs,pa,F=matrix(DATA/'species_function_matrix_filtered.csv')
    assert set(sp)==set(fs)
    F=F[[fs.index(s) for s in sp]]
    count=(A>0).sum(1).astype(float)
    L=np.log1p(F);valid=F>0
    cnt=(valid*count[:,None]).sum(0)
    mu=(L*count[:,None]).sum(0)/np.maximum(cnt,1)
    sd=np.sqrt((((L-mu)**2)*valid*count[:,None]).sum(0)/np.maximum(cnt,1))
    sd[sd<=0]=1
    Z=np.where(valid,(L-mu)/sd,0).astype('float32')
    np.savez_compressed(ROOT/'results'/'reconstructed_scaler.npz',mu=mu,sigma=sd,counts=cnt)
    return sp,sa,pa,A,F,Z

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn=torch.nn.MultiheadAttention(64,4,batch_first=True)
        self.linear1=torch.nn.Linear(64,256,bias=False)
        self.linear2=torch.nn.Linear(256,64,bias=False)
    def forward(self,x,counts=None):
        b,l,d=x.shape
        qkv=torch.nn.functional.linear(x,self.self_attn.in_proj_weight,self.self_attn.in_proj_bias)
        q,k,v=[z.reshape(b,l,4,16).transpose(1,2) for z in qkv.chunk(3,dim=-1)]
        a=q@k.transpose(-2,-1)/4
        if counts is not None:
            lc=counts.log()
            a=a+(lc[None,None,None,:] if counts.ndim==1 else lc[:,None,None,:])
        h=(a.softmax(-1)@v).transpose(1,2).reshape(b,l,d)
        h=self.self_attn.out_proj(h)
        x=torch.nn.functional.layer_norm(x+h,(64,))
        return torch.nn.functional.layer_norm(x+self.linear2(torch.relu(self.linear1(x))),(64,))

class Frozen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection=torch.nn.Linear(463,64,bias=False)
        self.encoder_layers=torch.nn.ModuleList([Encoder(),Encoder()])
        self.output_layer=torch.nn.Linear(64,1,bias=False)
        self.register_buffer('positional_encoding',torch.zeros(1,1090,64))
    def forward(self,x,counts=None):
        x=self.input_projection(x)
        for layer in self.encoder_layers:x=layer(x,counts)
        logits=self.output_layer(x).squeeze(-1)
        if counts is not None:logits=logits+(counts.log()[None,:] if counts.ndim==1 else counts.log())
        return logits.softmax(-1)
    def original_forward(self,x):
        x=self.input_projection(x)
        for layer in self.encoder_layers:
            h,_=layer.self_attn(x,x,x,need_weights=True)
            x=torch.nn.functional.layer_norm(x+h,(64,))
            x=torch.nn.functional.layer_norm(x+layer.linear2(torch.relu(layer.linear1(x))),(64,))
        return self.output_layer(x).squeeze(-1).softmax(-1)

def load_model():
    net=Frozen()
    state=torch.load(DATA/'final_full_model.pth',map_location='cpu',weights_only=True)
    net.load_state_dict(state,strict=True);net.eval()
    for p in net.parameters():p.requires_grad_(False)
    return net

def compressed_input(g,A,Z):
    ids=np.flatnonzero(A[:,g]>0);absent=A.shape[0]-len(ids)
    x=Z[ids].copy();count=np.ones(len(ids),dtype='float32')
    if absent:x=np.concatenate([x,np.zeros((1,Z.shape[1]),dtype='float32')]);count=np.r_[count,float(absent)].astype('float32')
    return ids,torch.from_numpy(x),torch.from_numpy(count)

@torch.inference_mode()
def forward_batches(net,x,counts=None,bs=32):
    return torch.cat([net(x[i:i+bs],counts) for i in range(0,len(x),bs)]).numpy()

def bc(a,b):return np.abs(a-b).sum(-1)/(np.abs(a+b).sum(-1)+1e-12)

class DeletionGame:
    def __init__(self,net,x,counts,k,players,baseline=0.0):
        self.net=net;self.x=x;self.counts=counts;self.k=k;self.players=np.asarray(players,dtype=int);self.baseline=baseline
        self.m=len(players);self.p0=forward_batches(net,x[None],counts)[0];self.calls=0
    def __call__(self,masks,return_predictions=False):
        masks=np.asarray(masks,dtype='float32')
        if masks.ndim==1:masks=masks[None]
        vals=[];pred=[]
        for start in range(0,len(masks),32):
            chunk=masks[start:start+32];xx=self.x[None].repeat(len(chunk),1,1)
            old=self.x[self.players,self.k]
            xx[:,self.players,self.k]=old[None]*(1-torch.from_numpy(chunk))+self.baseline*torch.from_numpy(chunk)
            pp=forward_batches(self.net,xx,self.counts);self.calls+=len(chunk)
            vals.extend(bc(pp,self.p0));pred.extend(pp)
        return np.array(pred) if return_predictions else np.array(vals)

def enumerate_masks(m):return ((np.arange(2**m)[:,None]>>np.arange(m))&1).astype('float32')

def exact_shapley(values,m):
    import math
    phi=np.zeros(m);ix=np.arange(2**m)
    size=np.array([int(i).bit_count() for i in ix])
    for j in range(m):
        left=ix[(ix&(1<<j))==0]
        weights=np.array([1/(m*math.comb(m-1,int(size[i]))) for i in left])
        phi[j]=weights@(values[left|(1<<j)]-values[left])
    return phi

def mc_shapley(game,m,nperm,seed):
    rng=np.random.default_rng(seed);perms=[]
    for _ in range((nperm+1)//2):
        p=rng.permutation(m);perms.extend([p,p[::-1]])
    perms=perms[:nperm];masks=np.zeros((nperm*(m+1),m),dtype='float32')
    for t,p in enumerate(perms):
        for j in range(m):masks[t*(m+1)+j+1]=masks[t*(m+1)+j];masks[t*(m+1)+j+1,p[j]]=1
    v=game(masks).reshape(nperm,m+1);phi=np.zeros((nperm,m))
    for t,p in enumerate(perms):phi[t,p]=np.diff(v[t])
    independent=phi.reshape(nperm//2,2,m).mean(1) if nperm%2==0 else phi
    return phi.mean(0),independent.std(0,ddof=1)/np.sqrt(len(independent))

def endpoint(game,m):
    masks=np.r_[np.eye(m),1-np.eye(m),np.ones((1,m))]
    v=game(masks);return .5*(v[:m]+v[-1]-v[m:2*m]),v[:m],v[-1]
