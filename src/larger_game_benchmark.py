from cv_shapley import *
from kernel_baseline import kernel_shap
m=12;mm=enumerate_masks(m);rows=[]
for family in ['heterogeneous','redundant','cooperative','compensatory']:
    for seed in range(101,111):
        rng=np.random.default_rng(seed);base=rng.normal(0,.4,12);B=rng.normal(0,.35,(m,12));B[6:]=0
        if family=='redundant':B[:6]=B[0]
        if family=='compensatory':B[1]=-B[0];B[3]=-B[2];B[5]=-B[4]
        lp=base+mm@B
        if family=='cooperative':lp+=(mm[:,0]*mm[:,1])[:,None]*B[0]*3+(mm[:,2]*mm[:,3])[:,None]*B[2]*2
        p=np.exp(lp-lp.max(1,keepdims=True));p/=p.sum(1,keepdims=True);v=bc(p,p[0]);truth=exact_shapley(v,m);den=max(np.abs(truth).sum(),1e-12)
        lookup=lambda z:v[(np.asarray(z,dtype=int)*(1<<np.arange(m))).sum(1)]
        np.savez_compressed(ROOT/'results'/f'larger_{family}_seed{seed}.npz',base=base,B=B,exact=truth,values=v)
        for budget in [16,32,64]:
            for rep in range(10):
                ss=9900+rep*71+seed;a,_=mc_shapley(lookup,m,budget,ss);b,_,meta=cv_shapley(lookup,m,budget,ss);k=kernel_shap(lookup,m,budget*(m+1),ss)
                for name,val in [('antithetic_mc',a),('probe_cubic_cv',b),('kernel_shap',k)]:rows.append({'players':m,'family':family,'seed':seed,'replicate':rep,'budget_permutations':budget,'method':name,'normalized_l1_error':float(np.abs(val-truth).sum()/den),'squared_error':float(np.sum((val-truth)**2)),'dummy_mass_fraction':float(np.abs(val[6:]).sum()/max(np.abs(val).sum(),1e-12))})
        print(family,seed,'larger game complete',flush=True)
table('results/larger_game_matched_budget.csv',rows)
# Small-game exhaustive KernelSHAP check at a budget that permits enumeration.
checks=[]
for path in sorted((ROOT/'results').glob('controlled_*_seed*.npz')):
    d=np.load(path);vv=bc(d['truth_predictions'],d['truth_predictions'][0]);look=lambda z:vv[(np.asarray(z,dtype=int)*(1<<np.arange(8))).sum(1)]
    k=kernel_shap(look,8,288,99);checks.append({'case':path.stem,'queries':256,'exact_kernel_error':float(np.max(np.abs(k-d['exact_truth'])))})
table('results/kernel_exact_small_game_checks.csv',checks)
print('kernel exhaustive error',max(r['exact_kernel_error'] for r in checks),flush=True)
