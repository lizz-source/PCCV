from cv_shapley import *
cfg=json.loads((ROOT/'config.json').read_text());rows=[]
for family in ['heterogeneous','redundant','cooperative','compensatory','null']:
    for seed in cfg['controlled_benchmark_seeds']:
        if family=='null':
            m=8;v=np.zeros(2**m);truth=np.zeros(m)
        else:
            d=np.load(ROOT/'results'/f'controlled_{family}_seed{seed}.npz');m=d['masks'].shape[1];v=bc(d['truth_predictions'],d['truth_predictions'][0]);truth=d['exact_truth']
        lookup=lambda xx:v[(np.asarray(xx,dtype=int)*(1<<np.arange(m))).sum(1)]
        den=max(np.abs(truth).sum(),1e-12)
        for budget in [16,32,64]:
            for rep in range(10):
                s=7000+rep*71+seed
                a,_=mc_shapley(lookup,m,budget,s);b,_,meta=cv_shapley(lookup,m,budget,s)
                for name,val,queries in [('antithetic_mc',a,budget*(m+1)),('probe_cubic_cv',b,meta['theoretical_query_budget'])]:
                    rows.append({'family':family,'seed':seed,'replicate':rep,'budget_permutations':budget,'method':name,'queries_upper_bound':queries,'normalized_l1_error':float(np.abs(val-truth).sum()/den),'squared_error':float(np.sum((val-truth)**2)),'efficiency_abs_error':float(abs(val.sum()-v[-1])),'dummy_abs_mass':float(np.abs(val[4:]).sum())})
table('results/cv_matched_budget_evaluation.csv',rows)
summary=[]
for family in ['heterogeneous','redundant','cooperative','compensatory','null']:
    for budget in [16,32,64]:
        for method in ['antithetic_mc','probe_cubic_cv']:
            rr=[r for r in rows if r['family']==family and r['budget_permutations']==budget and r['method']==method]
            summary.append({'family':family,'budget_permutations':budget,'method':method,'n_cases':len(rr),'median_nl1_error':float(np.median([r['normalized_l1_error'] for r in rr])),'mean_squared_error':float(np.mean([r['squared_error'] for r in rr])),'max_efficiency_error':max(r['efficiency_abs_error'] for r in rr),'max_dummy_abs_mass':max(r['dummy_abs_mass'] for r in rr)})
table('results/cv_matched_budget_summary.csv',summary)
print([r for r in summary if r['budget_permutations']==32],flush=True)
