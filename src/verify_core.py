from common import *
import copy,itertools
sp,sa,pa,A,F,Z=load_data();net=load_model();rng=np.random.default_rng(20261027)
rich=(A>0).sum(0);idx=np.argsort(rich);chosen=[]
for block in np.array_split(idx,3):chosen.extend(rng.choice(block,6,replace=False).tolist())
rows=[]
with torch.inference_mode():
    for g in chosen:
        ids,x,c=compressed_input(g,A,Z);dense=torch.zeros((1,1090,463));dense[0,ids]=x[:len(ids)]
        k=pa.index('ko02020');x[:,k]=0;dense[:,:,k]=0
        net.original_forward(dense);net(x[None],c);dt=[];ct=[]
        for _ in range(5):
            t=time.perf_counter();pd=net.original_forward(dense).numpy()[0];dt.append(time.perf_counter()-t)
            t=time.perf_counter();pc=net(x[None],c).numpy()[0];ct.append(time.perf_counter()-t)
        pp=np.empty(1090);pp[ids]=pc[:len(ids)];absent=np.ones(1090,bool);absent[ids]=False;pp[absent]=pc[-1]/absent.sum()
        perm=rng.permutation(len(c));rr=net(x[perm][None],c[perm]).numpy()[0];rr=rr[np.argsort(perm)]
        rows.append({'sample_id':sa[g],'richness':len(ids),'max_probability_error':float(np.max(np.abs(pd-pp))),'permutation_equivariance_error':float(np.max(np.abs(pc-rr))),'dense_median_seconds':float(np.median(dt)),'compressed_median_seconds':float(np.median(ct)),'speedup':float(np.median(dt)/np.median(ct)),'attention_cell_ratio':float(1090**2/(len(ids)+1)**2)})
table('results/compression_benchmark.csv',rows)
# Compare full and collapsed parameter gradients in double precision.
a=Frozen().double();a.load_state_dict(net.state_dict());b=copy.deepcopy(a)
x=torch.zeros((1,48,463),dtype=torch.float64);x[0,:7]=torch.tensor(Z[:7],dtype=torch.float64)
y=torch.zeros((1,48),dtype=torch.float64);y[0,:7]=torch.softmax(torch.arange(7,dtype=torch.float64),dim=0)
pp=a.original_forward(x);loss=(pp-y).abs().sum()/2;loss.backward()
xx=torch.cat([x[:,:7],x[:,7:8]],1);yy=torch.cat([y[:,:7],y[:,7:].sum(1,keepdim=True)],1);c=torch.tensor([1]*7+[41],dtype=torch.float64)
qq=b(xx,c);loss2=(qq-yy).abs().sum()/2;loss2.backward()
grad=max(float((p.grad-q.grad).abs().max()) for p,q in zip(a.parameters(),b.parameters()) if p.grad is not None)
# Independently enumerate all permutations for a 4-player interaction game.
mm=enumerate_masks(4);v=mm@np.array([.1,-.2,.5,0])+2*mm[:,0]*mm[:,1]+3*mm[:,0]*mm[:,2]*mm[:,3]
ph=exact_shapley(v,4);brute=np.zeros(4)
for p in itertools.permutations(range(4)):
    ix=0
    for j in p:brute[j]+=v[ix|(1<<j)]-v[ix];ix|=1<<j
brute/=24
ver={'compression_probability_max_error':max(r['max_probability_error'] for r in rows),'equivariance_max_error':max(r['permutation_equivariance_error'] for r in rows),'gradient_max_abs_error_float64':grad,'gradient_loss_difference':float(abs(loss.item()-loss2.item())),'exact_shapley_vs_permutation_max_error':float(np.max(np.abs(ph-brute))),'speedup_median':float(np.median([r['speedup'] for r in rows])),'speedup_min':min(r['speedup'] for r in rows),'speedup_max':max(r['speedup'] for r in rows)}
ver['pass']=ver['compression_probability_max_error']<1e-5 and ver['equivariance_max_error']<1e-5 and grad<1e-9 and ver['exact_shapley_vs_permutation_max_error']<1e-12
from cv_shapley import cv_shapley
# All endpoint probes vanish, but intermediate coalitions have an effect.
# The sixth player is a true null player and must remain so in the estimator.
dm=enumerate_masks(6);dv=(dm[:,:5].sum(1)==2).astype(float)
lookup=lambda z:dv[(np.asarray(z,dtype=int)*(1<<np.arange(6))).sum(1)]
dummy_error=0.0
for seed in range(10):
    result,_,_=cv_shapley(lookup,6,16,seed);dummy_error=max(dummy_error,abs(float(result[-1])))
ver['zero_endpoint_higher_order_dummy_error']=dummy_error;ver['pass']=ver['pass'] and dummy_error<1e-12
dump('results/core_verification.json',ver);print(ver,flush=True);assert ver['pass']
