from cv_shapley import *
cfg=json.loads((ROOT/'config.json').read_text());rows=[]
for family in ['heterogeneous','redundant','cooperative','compensatory']:
    for seed in cfg['controlled_benchmark_seeds']:
        dd=np.load(ROOT/'results'/f'controlled_{family}_seed{seed}.npz');m=8;v=bc(dd['truth_predictions'],dd['truth_predictions'][0]);truth=dd['exact_truth'];den=max(np.abs(truth).sum(),1e-12)
        look=lambda xx:v[(np.asarray(xx,dtype=int)*(1<<np.arange(m))).sum(1)]
        for rep in range(10):
            s=7000+rep*71+seed;quad,_,meta=cv_shapley(look,m,32,s,degree=2);matched,_=mc_shapley(look,m,28,s)
            rows.append({'family':family,'seed':seed,'replicate':rep,'method':'probe_quadratic_cv','normalized_l1_error':float(np.abs(quad-truth).sum()/den),'max_difference_from_same_permutations_without_cv':float(np.max(np.abs(quad-matched))),'queries_upper_bound':meta['theoretical_query_budget']})
table('results/quadratic_control_ablation.csv',rows);print('max quadratic correction',max(r['max_difference_from_same_permutations_without_cv'] for r in rows),flush=True)
