"""Post hoc, frozen-method comparison with a recent polynomial-regression baseline."""
from poly_baseline import *
from kernel_baseline import kernel_shap
rows=[];checks=[]
# Independent low-order test, plus exact-enumeration agreement with KernelSHAP.
m=6;z=enumerate_masks(m);rng=np.random.default_rng(6327);v=z@rng.normal(size=m)+2*z[:,0]*z[:,1]-3*z[:,2]*z[:,3]*z[:,4]
game=lambda masks:v[(np.asarray(masks,dtype=int)*(1<<np.arange(m))).sum(1)]
truth=exact_shapley(v,m)
for degree in [1,3]:
    val=poly_shapley(game,m,64,73,degree=degree)
    reference=truth if degree==3 else kernel_shap(game,m,64,73)
    err=float(np.max(np.abs(val-reference)));checks.append({'check':f'degree_{degree}_enumeration','max_error':err,'pass':err<1e-8})
assert all(c['pass'] for c in checks),checks
for file in sorted((ROOT/'results').glob('larger_*_seed*.npz')):
    _,family,ss=file.stem.split('_');seed=int(ss.removeprefix('seed'));d=np.load(file);v=d['values'];truth=d['exact'];m=12;den=max(np.abs(truth).sum(),1e-12)
    game=lambda masks:v[(np.asarray(masks,dtype=int)*(1<<np.arange(m))).sum(1)]
    for budget in [32,64]:
        for rep in range(10):
            for pairing in [False,True]:
                val=poly_shapley(game,m,budget*(m+1),9900+rep*71+seed,degree=3,pairing=pairing)
                rows.append({'players':m,'family':family,'seed':seed,'replicate':rep,'budget_permutations':budget,'method':'poly_shap_degree3_paired' if pairing else 'poly_shap_degree3_unpaired','normalized_l1_error':float(np.abs(val-truth).sum()/den),'squared_error':float(np.sum((val-truth)**2)),'dummy_mass_fraction':float(np.abs(val[6:]).sum()/max(np.abs(val).sum(),1e-12)),'efficiency_error':float(abs(val.sum()-v[-1]))})
    print(file.stem,'PolySHAP complete',flush=True)
table('results/poly_shap_matched_budget.csv',rows)
dump('results/poly_baseline_verification.json',{'checks':checks,'all_pass':all(c['pass'] for c in checks),'cases':40,'rows':len(rows),'degree':3,'frontier_terms':298,'budgets_permutations':[32,64],'sampling':'kernel proportional; both paired and unpaired','limitations':'Independent Algorithm 1 implementation, not official full package; leverage sampling and adaptive frontier selection are not evaluated.'})
print('PolySHAP rows',len(rows),flush=True)
