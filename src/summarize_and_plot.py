from report_common import *
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220,'svg.fonttype':'none'})
R=ROOT/'results';figdir=ROOT/'figures';figdir.mkdir(exist_ok=True)
def read(n):return pd.read_csv(R/n,keep_default_na=False,float_precision='round_trip')
def save(fig,name):
    fig.savefig(figdir/(name+'.png'),bbox_inches='tight');fig.savefig(figdir/(name+'.svg'),bbox_inches='tight');plt.close(fig)
def finite(x):
    x=np.asarray(x,dtype=float);return x[np.isfinite(x)]
def median(x):return float(np.median(finite(x)))
methods=['proportional','abundance','uniform','single_deletion','endpoint_average','antithetic_mc','probe_cubic_cv']
labels={'proportional':'Carrier allocation','abundance':'Abundance','uniform':'Uniform','single_deletion':'Single deletion','endpoint_average':'Endpoint mean','antithetic_mc':'Permutation Shapley','probe_cubic_cv':'PCCV'}
colors={'proportional':'#a05d56','abundance':'#aa8866','uniform':'#999999','single_deletion':'#c9a227','endpoint_average':'#519591','antithetic_mc':'#536b9c','probe_cubic_cv':'#203d6a'}
assert json.loads((R/'real_evaluation_complete.json').read_text())['cases']==108
models=read('model_test_summary.csv');assert len(models)==6
cv=read('cv_matched_budget_evaluation.csv');controlled=read('controlled_attribution_metrics.csv');fits=read('controlled_predictor_fits.csv')
real=read('real_attribution_metrics.csv');conv=read('real_sampling_convergence.csv');exact=read('real_exact_subgames.csv');compression=read('compression_benchmark.csv')
meta=json.loads((R/'metadata_audit.json').read_text());hold=json.loads((ROOT/'study_holdout/results/annotation_seed11.json').read_text());hdesign=json.loads((ROOT/'study_holdout/results/holdout_design.json').read_text())
summary={'data':{'samples':2894,'species':1090,'pathways':463,'train':2025,'validation':434,'test':435},'metadata':meta,'compression':json.loads((R/'core_verification.json').read_text()),'models':models.to_dict('records'),'study_holdout':{**hdesign,**hold},'controlled_predictor_test_bc_median':median(fits['test_bc']),'controlled_exact_model_vs_oracle_nl1_median':median(fits['exact_model_vs_mechanism_normalized_l1']),'real_cases':int(real['case_id'].nunique()),'real_exact_subgames':int(exact['case_id'].nunique())}
cv_summary=[]
for family in ['heterogeneous','redundant','cooperative','compensatory']:
    rr=cv[(cv.family==family)&(cv.budget_permutations==32)]
    a=rr[rr.method=='antithetic_mc'];b=rr[rr.method=='probe_cubic_cv']
    cv_summary.append({'family':family,'mc_median_nl1':median(a.normalized_l1_error),'pccv_median_nl1':median(b.normalized_l1_error),'mse_ratio_pccv_over_mc':float(b.squared_error.mean()/a.squared_error.mean())})
summary['controlled_query_budget_32']=cv_summary
real_summary=[]
for method in methods:
    d=real[real.method==method]
    real_summary.append({'method':method,'median_rank_rho_vs_mc':median(d.rank_rho_vs_mc) if method!='uniform' else None,'median_nl1_vs_mc':median(d.nl1_difference_vs_mc),'mean_deletion_auc':float(d.relative_deletion_auc.mean()),'mean_restoration_auc':float(d.relative_insertion_auc.mean()),'median_random_coalition_mse':median(d.random_coalition_mse),'max_efficiency_error':float(d.efficiency_abs_error.max())})
summary['real_method_metrics']=real_summary
summary['sampling_replicate_difference']={'mc_median_nl1':median(conv.mc_replicate_nl1_difference),'pccv_median_nl1':median(conv.cv_replicate_nl1_difference),'mc_median_rho':median(conv.mc_replicate_rho),'pccv_median_rho':median(conv.cv_replicate_rho)}
prior=read('presence_conditioned_prior_test.csv');hprior=pd.read_csv(ROOT/'study_holdout/results/presence_conditioned_prior_test.csv')
summary['presence_conditioned_prior']={'internal_test_bc':float(prior.bc.mean()),'internal_test_jsd':float(prior.jsd.mean()),'study_holdout_test_bc':float(hprior.bc.mean()),'study_holdout_test_jsd':float(hprior.jsd.mean())}
cross=read('real_cross_model_endpoints.csv');crossrows=[]
for a,b in [(11,22),(11,33),(22,33)]:
    vals=[]
    for _,d in cross.groupby('case_id'):
        w=d.pivot(index='species_id',columns='training_seed',values='endpoint_attribution')
        if np.std(w[a])>1e-15 and np.std(w[b])>1e-15:vals.append(float(spearmanr(w[a],w[b]).statistic))
    crossrows.append({'training_seed_a':a,'training_seed_b':b,'cases':len(vals),'median_endpoint_rank_rho':median(vals)})
summary['cross_training_seed_endpoint_agreement']=crossrows
summary['real_exact_errors']={method:median(exact[exact.method==method].normalized_l1_error) for method in ['proportional','antithetic_mc','probe_cubic_cv']}
larger=read('larger_game_matched_budget.csv');larger_summary=[]
for family in ['heterogeneous','redundant','cooperative','compensatory']:
    for method in ['antithetic_mc','probe_cubic_cv','kernel_shap']:
        d=larger[(larger.family==family)&(larger.method==method)&(larger.budget_permutations==32)]
        larger_summary.append({'family':family,'method':method,'median_normalized_l1_error':median(d.normalized_l1_error),'mean_squared_error':float(d.squared_error.mean()),'median_dummy_mass_fraction':median(d.dummy_mass_fraction)})
summary['larger_game_12_players_budget32']=larger_summary
poly=read('poly_shap_matched_budget.csv');poly_summary=[]
for (family,budget,method),d in poly.groupby(['family','budget_permutations','method']):
    poly_summary.append({'family':family,'budget_permutations':int(budget),'method':method,'median_normalized_l1_error':median(d.normalized_l1_error),'mean_squared_error':float(d.squared_error.mean()),'median_dummy_mass_fraction':median(d.dummy_mass_fraction)})
summary['poly_shap_supplementary_comparison']=poly_summary
summary['identifiability_demo']=json.loads((R/'identifiability_verification.json').read_text())
quad=read('quadratic_control_ablation.csv');summary['quadratic_correction_max_abs']=float(quad.max_difference_from_same_permutations_without_cv.max())
hd=read('study_holdout_explanation_audit.csv');summary['study_holdout_explanation_audit']={method:{'n_cases':len(d),'mean_deletion_auc':float(d.deletion_auc.mean()),'mean_restoration_auc':float(d.restoration_auc.mean())} for method,d in hd.groupby('method')}
# Cluster the six pathway comparisons by sampled community. These confidence
# intervals describe the sampled model audit and are not ecological validation.
bootstrap=[];rng=np.random.default_rng(2525)
for target,reference in [('antithetic_mc','proportional'),('probe_cubic_cv','proportional'),('probe_cubic_cv','antithetic_mc')]:
    for metric in ['relative_deletion_auc','relative_insertion_auc']:
        wide=real.pivot(index=['sample_id','pathway_id'],columns='method',values=metric)
        difference=(wide[target]-wide[reference]).groupby(level='sample_id').mean().to_numpy()
        boot=difference[rng.integers(0,len(difference),(10000,len(difference)))].mean(1)
        bootstrap.append({'comparison':target+' minus '+reference,'metric':metric,'n_sample_clusters':len(difference),'mean_difference':float(difference.mean()),'bootstrap_q025':float(np.quantile(boot,.025)),'bootstrap_q975':float(np.quantile(boot,.975))})
summary['paired_sample_cluster_bootstrap']=bootstrap;table('results/paired_cluster_bootstrap.csv',bootstrap)
if (R/'real_sensitivity_checks.csv').exists():
    sens=read('real_sensitivity_checks.csv');summary['sensitivity']={kind:{'n':len(d),'median_rho':median(d.rank_rho_vs_primary_bc_zero_baseline)} for kind,d in sens.groupby('check')}
table('results/real_method_summary.csv',real_summary);dump('results/final_scientific_summary.json',summary)

fig,ax=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
ax[0].scatter(compression.richness,compression.max_probability_error,color='#203d6a');ax[0].set(xlabel='Present species',ylabel='Maximum probability error',yscale='log',title='A  Dense versus collapsed inference');ax[0].axhline(1e-5,color='#a05d56',ls='--',label='Tolerance');ax[0].legend(frameon=False)
ax[1].scatter(compression.richness,compression.speedup,color='#203d6a');ax[1].set(xlabel='Present species',ylabel='Measured speedup',title='B  CPU inference, median of five repeats')
save(fig,'Figure1_exact_sparse_inference')

families=['heterogeneous','redundant','cooperative','compensatory'];show=['proportional','single_deletion','endpoint_average','permutation_shapley_32']
fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained');x=np.arange(4)
for j,method in enumerate(show):
    val=[median(controlled[(controlled.family==f)&(controlled.method==method)].normalized_l1_error) for f in families]
    ax[0].bar(x+(j-1.5)*.2,val,.19,label=method.replace('_',' '))
ax[0].set_xticks(x,families,rotation=15);ax[0].set(ylabel='Normalized L1 error (symmetric log scale)',title='A  Exact coalition reference');ax[0].set_yscale('symlog',linthresh=.05);ax[0].set_yticks([0,.05,.5,5,25],['0','0.05','0.5','5','25']);ax[0].legend(fontsize=8,frameon=False)
dummy=controlled[controlled.family!='null'].groupby('method').dummy_mass_fraction.median();aa=[v for v in show if v in dummy.index]
ax[1].bar(range(len(aa)),[dummy[v] for v in aa],color=['#a05d56','#c9a227','#519591','#536b9c']);ax[1].set_xticks(range(len(aa)),['Allocation','Single','Endpoint','Shapley']);ax[1].set(ylabel='Fraction of absolute mass on null carriers',title='B  Planted null carriers')
save(fig,'Figure2_ground_truth_attribution')

fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for ax,family in zip(axs.flat,families):
    for method,col in [('antithetic_mc','#536b9c'),('probe_cubic_cv','#203d6a')]:
        ss=[cv[(cv.family==family)&(cv.method==method)&(cv.budget_permutations==b)] for b in [16,32,64]]
        ax.plot([16,32,64],[median(d.normalized_l1_error) for d in ss],marker='o',color=col,label=labels[method])
    ax.set(title=family,xlabel='Nominal permutation budget',ylabel='Median normalized L1 error');ax.legend(frameon=False,fontsize=8)
save(fig,'Figure3_matched_budget_estimation')

fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for j,rep in enumerate(['annotation','fixed_shuffled_annotation']):
    d=models[models.representation==rep]
    ax[0].bar(j,d.test_jsd.mean(),yerr=d.test_jsd.std(),color=['#203d6a','#a05d56'][j],capsize=4)
prior=read('presence_conditioned_prior_test.csv');ax[0].bar(2,prior.jsd.mean(),color='#999999');ax[0].set_xticks([0,1,2],['Annotation','Shuffled mapping','Conditional prior']);ax[0].set(ylabel='Test Jensen–Shannon divergence',title='A  Resource-internal split, three seeds')
hprior=pd.read_csv(ROOT/'study_holdout/results/presence_conditioned_prior_test.csv');ax[1].bar([0,1],[hold['test_jsd'],hprior.jsd.mean()],color=['#203d6a','#999999']);ax[1].set_xticks([0,1],['Annotation','Conditional prior']);ax[1].set(ylabel='Test Jensen–Shannon divergence',title='B  Entire-study holdout (one seed)')
save(fig,'Figure4_predictor_controls')

curves=read('real_deletion_curves.csv');fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for method in ['proportional','single_deletion','endpoint_average','antithetic_mc','probe_cubic_cv']:
    d=curves[curves.method==method].copy();d['d']=d.deletion_bc/d.joint_bc;d['i']=d.restoration_gain_bc/d.joint_bc
    g=d.groupby('fraction')[['d','i']].mean();ax[0].plot(g.index,g.d,label=labels[method],color=colors[method]);ax[1].plot(g.index,g.i,label=labels[method],color=colors[method])
ax[0].set(xlabel='Fraction of carriers deleted',ylabel='Response / joint response',title='A  Deletion audit');ax[1].set(xlabel='Fraction of carriers restored',ylabel='Restoration gain / joint response',title='B  Restoration audit');ax[0].legend(frameon=False,fontsize=8)
save(fig,'Figure5_real_perturbation_faithfulness')

fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for j,method in enumerate(['proportional','antithetic_mc','probe_cubic_cv']):
    vals=exact[exact.method==method].groupby('case_id').normalized_l1_error.median().to_numpy();ax[0].scatter(np.repeat(j,len(vals)),vals,color=colors[method])
ax[0].set_xticks([0,1,2],['Allocation','Shapley','PCCV']);ax[0].set(ylabel='Normalized L1 error',title='A  Six exact conditional subgames')
ax[1].scatter(conv.mc_replicate_nl1_difference,conv.cv_replicate_nl1_difference,color='#203d6a',alpha=.5,s=12);limit=max(conv.mc_replicate_nl1_difference.max(),conv.cv_replicate_nl1_difference.max());ax[1].plot([0,limit],[0,limit],ls='--',color='#999999');ax[1].set(xlabel='Shapley replicate disagreement',ylabel='PCCV replicate disagreement',title='B  Independent sampling runs')
save(fig,'Figure6_exactness_and_sampling')

attrs=read('real_attributions.csv');d=attrs[(attrs.method=='probe_cubic_cv')&attrs.case_id.str.endswith('p00')];wide=d.pivot(index='species_id',columns='case_id',values='attribution').fillna(0);top=wide.abs().mean(axis=1).nlargest(14).index;W=wide.loc[top].to_numpy();W=W/(np.abs(W).sum(0,keepdims=True)+1e-12)
fig,ax=plt.subplots(figsize=(11,6),layout='constrained');lim=np.max(np.abs(W));im=ax.imshow(W,aspect='auto',cmap='RdBu_r',vmin=-lim,vmax=lim);ax.set_yticks(range(len(top)),[s.replace('_',' ') for s in top],fontsize=8);ax.set_xticks(range(len(wide.columns)),wide.columns,rotation=90,fontsize=8);ax.set(title='Carrier attribution in the highest validation-response pathway',xlabel='Preselected held-out community');fig.colorbar(im,ax=ax,label='Signed fraction of displayed absolute attribution')
save(fig,'Figure7_carrier_context_profiles')

fig,axs=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
for ax,family in zip(axs.flat,['heterogeneous','redundant','cooperative','compensatory']):
    for method,col in [('antithetic_mc','#536b9c'),('probe_cubic_cv','#203d6a'),('kernel_shap','#a05d56')]:
        vals=[median(larger[(larger.family==family)&(larger.method==method)&(larger.budget_permutations==budget)].normalized_l1_error) for budget in [16,32,64]]
        ax.plot([16,32,64],vals,marker='o',label=labels.get(method,'KernelSHAP'),color=col)
    for method,col,lab in [('poly_shap_degree3_paired','#39836e','PolySHAP-3 paired'),('poly_shap_degree3_unpaired','#8a699b','PolySHAP-3 unpaired')]:
        vals=[median(poly[(poly.family==family)&(poly.method==method)&(poly.budget_permutations==b)].normalized_l1_error) for b in [32,64]]
        ax.plot([32,64],vals,marker='s',ls='--',color=col,label=lab)
    ax.set(title=family,xlabel='Nominal permutation-equivalent query budget',ylabel='Median normalized L1 error (log scale)',yscale='log');ax.legend(frameon=False,fontsize=7.2)
save(fig,'Figure8_larger_games_and_kernel_baseline')

captions=[
('Figure1_exact_sparse_inference','Exact repeated-token inference. Eighteen richness-stratified communities; error is against original dense attention; CPU times are medians of five repeats after warmup.','compression_benchmark.csv'),
('Figure2_ground_truth_attribution','Attribution against exact coalition values in controlled response mechanisms. Panel A uses a symmetric log scale with a linear region within 0.05. Null carriers have no effect on the response generator. These are explanation benchmarks, not biological communities.','controlled_attribution_metrics.csv'),
('Figure3_matched_budget_estimation','Matched-query comparison of antithetic Shapley sampling and PCCV. Ten independent mechanism seeds and ten sampling replicates per family/budget. The compensatory family is retained as a failure boundary.','cv_matched_budget_evaluation.csv'),
('Figure4_predictor_controls','Freshly trained composition predictors. Panel A uses a resource-internal sample split and contains known repeated-subject overlap; panel B holds out an entire study with subject-disjoint training/validation. Error bars in A show training-seed SD.','model_test_summary.csv; presence_conditioned_prior_test.csv; study_holdout/results/annotation_seed11.json; study_holdout/results/presence_conditioned_prior_test.csv'),
('Figure5_real_perturbation_faithfulness','Deletion and restoration curves in all preselected test cases, averaged after normalization by each case joint response. Curves may exceed one under nonmonotone effects. These are interventions on model inputs.','real_deletion_curves.csv'),
('Figure6_exactness_and_sampling','Six exact eight-carrier conditional subgames and independent sampling agreement across full carrier sets. Other carriers are unchanged in exact subgames.','real_exact_subgames.csv; real_sampling_convergence.csv'),
('Figure7_carrier_context_profiles','Signed PCCV contributions for the fourteen largest mean-absolute carriers in the selected highest-response pathway. These profiles describe model dependence and do not establish ecological keystoneness.','real_attributions.csv'),
('Figure8_larger_games_and_kernel_baseline','Twelve-carrier games with all 4096 coalitions providing exact references, using ten new mechanism seeds and ten sampling replicates. A log error scale displays all five estimators. KernelSHAP uses kernel-proportional coalition sampling. PolySHAP-3 is an independent Algorithm 1 implementation with all 298 terms and paired/unpaired kernel-proportional sampling; it is evaluated at 32/64 budgets, where the budget exceeds the number of terms. No leverage or adaptive-frontier implementation is evaluated.','larger_game_matched_budget.csv; kernel_exact_small_game_checks.csv; poly_shap_matched_budget.csv')]
table('figures/figure_source_map.csv',[{'figure':a,'caption':b,'source':c} for a,b,c in captions])
(figdir/'CAPTIONS.md').write_text('\n\n'.join(f'**{a}.** {b}' for a,b,c in captions),encoding='utf8')
print(json.dumps(summary,ensure_ascii=False,default=lambda v:v.item() if hasattr(v,'item') else str(v)),flush=True)
