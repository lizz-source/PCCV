from cv_shapley import *
m=8;mm=enumerate_masks(m);rng=np.random.default_rng(0)
# Verify the cubic closed form independently against all coalitions.
w=rng.random(m);w/=w.sum();q=(mm@w)**3
closed=w+.5*w*(w*w).sum()-.5*w*w
assert np.max(np.abs(closed-exact_shapley(q,m)))<1e-12
rows=[]
for family in ['softmax','saturation','interaction']:
    B=rng.normal(0,.4,(m,12));B[4:]=0;base=rng.normal(0,.4,12);lp=base+mm@B
    if family=='saturation':lp=base+np.tanh(mm@B)*2
    if family=='interaction':lp=lp+(mm[:,0]*mm[:,1]*mm[:,2])[:,None]*B[0]*3
    p=np.exp(lp-lp.max(1,keepdims=True));p/=p.sum(1,keepdims=True);v=bc(p,p[0]);truth=exact_shapley(v,m);den=np.abs(truth).sum()
    lookup=lambda x:v[(np.asarray(x,dtype=int)*(1<<np.arange(m))).sum(1)]
    for seed in range(40):
        a,_=mc_shapley(lookup,m,32,seed+100);b,_,meta=cv_shapley(lookup,m,32,seed+100)
        rows.append({'family':family,'seed':seed,'mc_nl1':float(np.abs(a-truth).sum()/den),'cv_nl1':float(np.abs(b-truth).sum()/den),'cv_efficiency_error':float(abs(b.sum()-v[-1])),'cv_dummy_mass':float(np.abs(b[4:]).sum())})
table('results/cv_development_seed0.csv',rows)
print({f:{k:float(np.median([r[k] for r in rows if r['family']==f])) for k in ['mc_nl1','cv_nl1','cv_efficiency_error','cv_dummy_mass']} for f in ['softmax','saturation','interaction']},flush=True)
