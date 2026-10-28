"""Publication figures derived from frozen results without model fitting."""
from pathlib import Path
import hashlib,json,os,sys,importlib.util
ROOT=Path(__file__).resolve().parent;FROZEN=ROOT.parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp'/'matplotlib'))
os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('OMP_NUM_THREADS','2')
if importlib.util.find_spec('matplotlib') is None:
    sys.path.insert(0,str(FROZEN/'vendor'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager,ticker
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

FIG=ROOT/'figures';DATA=ROOT/'figure_data';PREVIEW=ROOT/'figure_previews';CASE=ROOT/'case_study'
for d in [FIG,DATA,PREVIEW,CASE]:d.mkdir(exist_ok=True)
font_dir=Path(os.environ.get('PCCV_TNR_FONT_DIR','C:/Windows/Fonts'))
for file in ['times.ttf','timesbd.ttf','timesi.ttf','timesbi.ttf']:
    font_file=font_dir/file
    if font_file.is_file():font_manager.fontManager.addfont(str(font_file))
try:
    font_manager.findfont('Times New Roman',fallback_to_default=False)
except ValueError as exc:
    raise RuntimeError('Install Times New Roman or set PCCV_TNR_FONT_DIR to your licensed font directory.') from exc
plt.rcParams.update({'font.family':'Times New Roman','font.size':14,'axes.labelsize':14,
 'axes.titlesize':15,'xtick.labelsize':13,'ytick.labelsize':13,'legend.fontsize':12.5,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.8,
 'xtick.major.width':.8,'ytick.major.width':.8,'xtick.major.size':3.5,'ytick.major.size':3.5,
 'lines.linewidth':1.8,'lines.markersize':5.3,'savefig.dpi':600,'svg.fonttype':'none',
 'axes.unicode_minus':True,'mathtext.fontset':'custom','mathtext.rm':'Times New Roman',
 'mathtext.it':'Times New Roman:italic','mathtext.bf':'Times New Roman:bold',
 'figure.facecolor':'white','axes.facecolor':'white','text.color':'#20262D',
 'axes.labelcolor':'#20262D','xtick.color':'#36414B','ytick.color':'#36414B'})
C={'probe_cubic_cv':'#214E76','antithetic_mc':'#25877F','proportional':'#B65F45',
   'single_deletion':'#C59B36','endpoint_average':'#8873A6','kernel_shap':'#A95170',
   'poly_shap_degree3_paired':'#6C8242','poly_shap_degree3_unpaired':'#7561A3',
   'prior':'#A3ADB4','abundance':'#718D99'}
N={'probe_cubic_cv':'PCCV','antithetic_mc':'Permutation sampling','proportional':'Annotation allocation',
   'single_deletion':'Single deletion','endpoint_average':'Endpoint mean','kernel_shap':'KernelSHAP',
   'poly_shap_degree3_paired':'PolySHAP 3 paired','poly_shap_degree3_unpaired':'PolySHAP 3 unpaired'}
FAMILIES=['heterogeneous','redundant','cooperative','compensatory'];records=[];sources={}
def read(name):
    path=FROZEN/name;sources[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path,float_precision='round_trip',keep_default_na=False)
def source_json(name):
    path=FROZEN/name;sources[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text(encoding='utf8'))
def panel(ax,letter,title):
    ax.set_title(title,loc='left',x=.075,pad=11)
    ax.annotate(letter,(0,1),xycoords='axes fraction',xytext=(0,11),textcoords='offset points',
                weight='bold',fontsize=17,ha='left',va='baseline')
def grid(ax):
    ax.set_axisbelow(True);ax.grid(axis='y',color='#E9EDF0',linewidth=.6)
def legend(fig,ax,ncol=3):
    fig.legend(*ax.get_legend_handles_labels(),loc='outside upper center',ncol=ncol,frameon=False,
               columnspacing=1.5,handlelength=2.2,handletextpad=.6)
def save(fig,name,data=None,source_names=()):
    records.append({'figure':name,'font':'Times New Roman','raster_dpi':600,
      'source_files':list(source_names),'derived_table':str(DATA/(name+'.csv')) if data is not None else None})
    only=set(sys.argv[sys.argv.index('--only')+1].split(',')) if '--only' in sys.argv else ({'Figure9_method_framework'} if '--framework-only' in sys.argv else None)
    if only is not None and name not in only:
        assert (FIG/(name+'.png')).is_file() and (FIG/(name+'.svg')).is_file()
        plt.close(fig);return
    fig.savefig(FIG/(name+'.png'),dpi=600,bbox_inches='tight',pad_inches=.08)
    fig.savefig(FIG/(name+'.svg'),bbox_inches='tight',pad_inches=.08)
    fig.savefig(PREVIEW/(name+'.png'),dpi=170,bbox_inches='tight',pad_inches=.08)
    if data is not None:data.to_csv(DATA/(name+'.csv'),index=False,float_format='%.17g')
    plt.close(fig)

compression=read('results/compression_benchmark.csv')
fig,ax=plt.subplots(1,2,figsize=(10.2,3.8),layout='constrained')
ax[0].scatter(compression.richness,compression.max_probability_error,color=C['probe_cubic_cv'],s=37,zorder=3)
ax[0].axhline(1e-5,color=C['proportional'],ls=(0,(4,3)),lw=1.3,label='Numerical tolerance')
ax[0].set(xlabel='Present species (count)',ylabel='Maximum probability difference',yscale='log')
ax[0].legend(frameon=False,loc='lower left',fontsize=12)
panel(ax[0],'A','Dense and grouped outputs');grid(ax[0])
ax[1].scatter(compression.richness,compression.speedup,color=C['probe_cubic_cv'],s=37,zorder=3)
ax[1].set(xlabel='Present species (count)',ylabel='CPU speedup (dense / grouped)')
panel(ax[1],'B','Inference at fixed parameters');grid(ax[1])
save(fig,'Figure1_exact_sparse_inference',compression,['results/compression_benchmark.csv'])

controlled=read('results/controlled_attribution_metrics.csv')
show=['proportional','single_deletion','endpoint_average','permutation_shapley_32']
canonical={'permutation_shapley_32':'antithetic_mc',**{x:x for x in show[:-1]}}
fig,ax=plt.subplots(1,2,figsize=(10.6,4),layout='constrained',gridspec_kw={'width_ratios':[1.2,1]})
x=np.arange(4);plotrows=[]
for j,method in enumerate(show):
    values=[controlled.loc[(controlled.family==family)&(controlled.method==method),'normalized_l1_error'].median() for family in FAMILIES]
    ax[0].bar(x+(j-1.5)*.19,values,.175,label=N[canonical[method]],color=C[canonical[method]])
    plotrows.extend({'panel':'A','family':fam,'method':method,'median_error':v} for fam,v in zip(FAMILIES,values))
ax[0].set_xticks(x,[s.capitalize() for s in FAMILIES],rotation=20,ha='right')
ax[0].set(ylabel='Median normalized L₁ error');ax[0].set_yscale('symlog',linthresh=.05)
ax[0].set_yticks([0,.05,.5,5,25],['0','0.05','0.5','5','25'])
panel(ax[0],'A','Accuracy against exact values');grid(ax[0])
dummy=controlled.loc[controlled.family!='null'].groupby('method').dummy_mass_fraction.median()
for j,method in enumerate(show):
    value=float(dummy[method]);ax[1].bar(j,value,.63,color=C[canonical[method]])
    if value<1e-12:ax[1].plot(j,0,'o',color=C[canonical[method]],ms=4,zorder=4)
    plotrows.append({'panel':'B','method':method,'null_mass_fraction':value})
ax[1].set_xticks(range(4),['Annotation\nallocation','Single\ndeletion','Endpoint\nmean','Permutation\nsampling'])
ax[1].tick_params(axis='x',labelsize=12)
ax[1].set(ylabel='Median null-carrier mass fraction',ylim=(-.018,max(float(dummy[m]) for m in show)*1.15))
panel(ax[1],'B','Planted null carriers');grid(ax[1]);legend(fig,ax[0],2)
save(fig,'Figure2_ground_truth_attribution',pd.DataFrame(plotrows),['results/controlled_attribution_metrics.csv'])

cv=read('results/cv_matched_budget_evaluation.csv')
fig,axs=plt.subplots(2,2,figsize=(10.2,7.4),layout='constrained');plotrows=[]
for k,(a,family) in enumerate(zip(axs.flat,FAMILIES)):
    for method in ['antithetic_mc','probe_cubic_cv']:
        values=[cv.loc[(cv.family==family)&(cv.method==method)&(cv.budget_permutations==budget),'normalized_l1_error'].median() for budget in [16,32,64]]
        a.plot([16,32,64],values,color=C[method],marker='o' if method=='probe_cubic_cv' else 's',
               ls='-' if method=='probe_cubic_cv' else '--',label=N[method])
        plotrows.extend({'family':family,'method':method,'budget_permutations':b,'median_error':v} for b,v in zip([16,32,64],values))
    a.set(xlabel='Nominal budget (permutation equivalents)',ylabel='Median normalized L₁ error',xticks=[16,32,64])
    panel(a,'ABCD'[k],family.capitalize());grid(a)
legend(fig,axs.flat[0],2)
save(fig,'Figure3_matched_budget_estimation',pd.DataFrame(plotrows),['results/cv_matched_budget_evaluation.csv'])

models=read('results/model_test_summary.csv');prior=read('results/presence_conditioned_prior_test.csv')
hold=source_json('study_holdout/results/annotation_seed11.json');hprior=read('study_holdout/results/presence_conditioned_prior_test.csv')
fig,ax=plt.subplots(1,2,figsize=(10.2,4),layout='constrained');plotrows=[]
for j,(rep,color) in enumerate([('annotation',C['probe_cubic_cv']),('fixed_shuffled_annotation',C['single_deletion'])]):
    d=models.loc[models.representation==rep,'test_jsd']
    ax[0].bar(j,d.mean(),.62,color=color,alpha=.87,yerr=d.std(),capsize=4,error_kw={'lw':1.2})
    ax[0].scatter(j+np.linspace(-.09,.09,len(d)),d,color='#20262D',s=15,zorder=4)
    plotrows.extend({'design':'Internal','representation':rep,'seed':int(row['seed']),'test_jsd':row['test_jsd']} for row in models.loc[models.representation==rep].to_dict('records'))
ax[0].bar(2,prior.jsd.mean(),.62,color=C['prior']);ax[0].set_xticks([0,1,2],['Annotation','Shuffled\nmapping','Conditional\nprior'])
ax[1].bar([0,1],[hold['test_jsd'],hprior.jsd.mean()],.62,color=[C['probe_cubic_cv'],C['prior']]);ax[1].set_xticks([0,1],['Annotation','Conditional prior'])
plotrows += [{'design':'Internal','representation':'conditional_prior','test_jsd':prior.jsd.mean()},
 {'design':'Study holdout','representation':'annotation','seed':11,'test_jsd':hold['test_jsd']},
 {'design':'Study holdout','representation':'conditional_prior','test_jsd':hprior.jsd.mean()}]
for a in ax:a.set(ylabel='Test Jensen–Shannon divergence',ylim=(0,.22));grid(a)
panel(ax[0],'A','Internal split (three seeds)');panel(ax[1],'B','Study holdout (one seed)')
save(fig,'Figure4_predictor_controls',pd.DataFrame(plotrows),['results/model_test_summary.csv','results/presence_conditioned_prior_test.csv','study_holdout/results/annotation_seed11.json','study_holdout/results/presence_conditioned_prior_test.csv'])

curves=read('results/real_deletion_curves.csv')
fig,ax=plt.subplots(1,2,figsize=(10.5,4),layout='constrained');plotrows=[]
for method in ['proportional','single_deletion','endpoint_average','antithetic_mc','probe_cubic_cv']:
    d=curves.loc[curves.method==method].copy();d['deletion_normalized']=d.deletion_bc/d.joint_bc;d['restoration_normalized']=d.restoration_gain_bc/d.joint_bc
    g=d.groupby('fraction')[['deletion_normalized','restoration_normalized']].mean()
    for j,column in enumerate(['deletion_normalized','restoration_normalized']):
        ax[j].plot(g.index,g[column],label=N[method],color=C[method],ls='--' if method=='antithetic_mc' else '-',marker='o' if method=='probe_cubic_cv' else None,ms=3.2)
    plotrows.extend({'method':method,'fraction':i,**row} for i,row in g.to_dict('index').items())
ax[0].set(xlabel='Fraction of carriers deleted',ylabel='Deletion response / joint response')
ax[1].set(xlabel='Fraction of carriers restored',ylabel='Restoration gain / joint response')
panel(ax[0],'A','Deletion audit');panel(ax[1],'B','Restoration audit')
for a in ax:a.set_xticks(np.linspace(0,1,6));grid(a)
legend(fig,ax[0],3)
save(fig,'Figure5_real_perturbation_faithfulness',pd.DataFrame(plotrows),['results/real_deletion_curves.csv'])

exact=read('results/real_exact_subgames.csv');conv=read('results/real_sampling_convergence.csv')
fig,ax=plt.subplots(1,2,figsize=(10.2,4),layout='constrained');plotrows=[]
for j,method in enumerate(['proportional','antithetic_mc','probe_cubic_cv']):
    d=exact.loc[exact.method==method].groupby('case_id').normalized_l1_error.median()
    ax[0].scatter(j+np.linspace(-.065,.065,len(d)),d,color=C[method],s=38,zorder=3)
    plotrows.extend({'method':method,'case_id':case,'median_error':v} for case,v in d.items())
ax[0].set_xticks([0,1,2],['Annotation\nallocation','Permutation\nsampling','PCCV']);ax[0].set(ylabel='Median normalized L₁ error')
ax[1].scatter(conv.mc_replicate_nl1_difference,conv.cv_replicate_nl1_difference,color=C['probe_cubic_cv'],alpha=.65,s=21,edgecolors='white',linewidths=.25)
limit=max(conv.mc_replicate_nl1_difference.max(),conv.cv_replicate_nl1_difference.max())*1.04
ax[1].plot([0,limit],[0,limit],ls='--',color=C['prior'],lw=1.2)
ax[1].set(xlabel='Permutation replicate difference',ylabel='PCCV replicate difference',xlim=(-.01,limit),ylim=(-.01,limit))
panel(ax[0],'A','Six exact conditional subgames');panel(ax[1],'B','Independent sampling runs')
for a in ax:grid(a)
save(fig,'Figure6_exactness_and_sampling',pd.DataFrame(plotrows),['results/real_exact_subgames.csv','results/real_sampling_convergence.csv'])
conv.to_csv(DATA/'Figure6_sampling_points.csv',index=False,float_format='%.17g')

attrs=read('results/real_attributions.csv');d=attrs.loc[(attrs.method=='probe_cubic_cv')&attrs.case_id.str.endswith('p00')]
wide=d.pivot(index='species_id',columns='case_id',values='attribution').fillna(0);top=wide.abs().mean(axis=1).nlargest(14).index
W=wide.loc[top].to_numpy();W=W/(np.abs(W).sum(axis=0,keepdims=True)+1e-12)
fig,ax=plt.subplots(figsize=(10.8,6.2),layout='constrained');lim=np.max(np.abs(W));im=ax.imshow(W,aspect='auto',cmap='RdBu_r',vmin=-lim,vmax=lim)
ax.set_yticks(range(len(top)),[s.replace('_',' ') for s in top],fontsize=14,fontstyle='italic')
ax.set_xticks(range(len(wide.columns)),wide.columns,rotation=60,ha='right',fontsize=12)
ax.set(xlabel='Preselected held-out community',title='Carrier contributions across community contexts')
for spine in ax.spines.values():spine.set_visible(False)
bar=fig.colorbar(im,ax=ax,shrink=.82,pad=.025);bar.set_label('Signed share of displayed absolute attribution',labelpad=10)
profile=pd.DataFrame(W,index=top,columns=wide.columns).rename_axis('species_id').reset_index()
save(fig,'Figure7_carrier_context_profiles',profile,['results/real_attributions.csv'])

larger=read('results/larger_game_matched_budget.csv');poly=read('results/poly_shap_matched_budget.csv')
fig,axs=plt.subplots(2,2,figsize=(10.6,7.6),layout='constrained');plotrows=[]
for k,(ax,family) in enumerate(zip(axs.flat,FAMILIES)):
    for method in ['antithetic_mc','probe_cubic_cv','kernel_shap','poly_shap_degree3_paired','poly_shap_degree3_unpaired']:
        data=poly if method.startswith('poly') else larger;budgets=[32,64] if method.startswith('poly') else [16,32,64]
        values=[data.loc[(data.family==family)&(data.method==method)&(data.budget_permutations==b),'normalized_l1_error'].median() for b in budgets]
        ax.plot(budgets,values,label=N[method],color=C[method],marker='s' if method.startswith('poly') else 'o',ls='--' if method.startswith('poly') or method=='antithetic_mc' else '-')
        plotrows.extend({'family':family,'method':method,'budget_permutations':b,'median_error':v} for b,v in zip(budgets,values))
    ax.set(xlabel='Nominal budget (permutation equivalents)',ylabel='Median normalized L₁ error',yscale='log',xticks=[16,32,64])
    ticks={'heterogeneous':[.04,.06,.1,.2],'redundant':[.002,.005,.01,.02],
           'cooperative':[.03,.04,.06,.1],'compensatory':[.3,.5,1,2,4]}[family]
    ax.set_yticks(ticks,[f'{v:g}' for v in ticks]);ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    grid(ax);panel(ax,'ABCD'[k],family.capitalize())
legend(fig,axs.flat[0],3)
save(fig,'Figure8_larger_games_and_kernel_baseline',pd.DataFrame(plotrows),['results/larger_game_matched_budget.csv','results/poly_shap_matched_budget.csv'])

# Example selection is fixed by the existing case plan, not the observed contrast.
plan=source_json('results/real_case_plan.json');sample=plan['test_sample_ids'][0];pathway=plan['pathways'][0]['id']
case=attrs.loc[(attrs.sample_id==sample)&(attrs.pathway_id==pathway)].copy();assert case.case_id.nunique()==1
pc=case.loc[case.method=='probe_cubic_cv'].sort_values(['observed_abundance','species_id'],ascending=[False,True]).copy()
pc['abundance_rank']=np.arange(1,len(pc)+1)
out=pc[['case_id','sample_id','pathway_id','species_id','observed_abundance','annotation_score','abundance_rank']].copy()
for method,prefix in [('probe_cubic_cv','pccv'),('antithetic_mc','permutation'),('proportional','allocation')]:
    part=case.loc[case.method==method].sort_values(['attribution','species_id'],ascending=[False,True]).copy();part[prefix+'_rank']=np.arange(1,len(part)+1)
    part=part.rename(columns={'attribution':prefix+'_attribution'})
    out=out.merge(part[['species_id',prefix+'_attribution',prefix+'_rank']],on='species_id',validate='one_to_one')
out['relative_abundance_percent']=out.observed_abundance*100
ab=out.sort_values(['observed_abundance','species_id'],ascending=[False,True]);rank=out.sort_values(['pccv_attribution','species_id'],ascending=[False,True])
display=list(dict.fromkeys(list(ab.species_id.head(3))+list(rank.species_id.head(3))+list(rank.species_id.tail(1))))
out['displayed_in_figure']=out.species_id.isin(display);out=out.sort_values('abundance_rank')
out.to_csv(CASE/'Case_Study_S1.csv',index=False,float_format='%.17g',encoding='utf-8-sig')
chosen=out.loc[out.displayed_in_figure].copy();assert len(out)==78 and len(chosen)==6
meta={'case_id':str(out.case_id.iloc[0]),'sample_id':sample,'pathway_id':pathway,'carriers':len(out),
 'selection':'First test sample and first pathway in frozen real_case_plan.json; no selection by attribution contrast.',
 'display_rule':'Union of top three abundance ranks, top three descending signed PCCV ranks, and minimum signed PCCV carrier.',
 'ranking':'Descending signed contribution or abundance; species identifier breaks ties; ranks use all 78 carriers.',
 'negative_pccv_carriers':int((out.pccv_attribution<0).sum()),'displayed_species':list(chosen.species_id),
 'estimates':'Archived averages of two sampling runs; no new fit, inference or statistical test.',
 'interpretation':'Model coordinate-ablation contributions, not measured activity or verified biological causation.',
 'source_sha256':{k:sources[k] for k in ['results/real_attributions.csv','results/real_case_plan.json']}}
(CASE/'case_study_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
fig,ax=plt.subplots(1,2,figsize=(10.8,5.0),layout='constrained',sharey=True,gridspec_kw={'width_ratios':[1,1.38]})
def taxon_label(species):
    if '_sp_' in species:
        genus,strain=species.split('_sp_',1)
        return r'$\mathit{'+genus+'}$ sp. '+strain.replace('_',' ')
    return r'$\mathit{'+species.replace('_',r'\ ')+'}$'
y=np.arange(len(chosen));names=[taxon_label(s) for s in chosen.species_id]
ax[0].barh(y,chosen.relative_abundance_percent,color=C['abundance'],height=.57)
ax[0].set_yticks(y,names,fontsize=13);ax[0].invert_yaxis();ax[0].set(xlabel='Relative abundance (%)',xlim=(0,14.4))
for yi,(_,row) in zip(y,chosen.iterrows()):
    ax[0].text(row.relative_abundance_percent+.2,yi,f'{row.relative_abundance_percent:.2f}  (#{int(row.abundance_rank)})',va='center',fontsize=13)
ax[1].barh(y,chosen.pccv_attribution*1000,color=C['probe_cubic_cv'],height=.57,label='PCCV')
ax[1].scatter(chosen.permutation_attribution*1000,y,s=44,facecolors='white',edgecolors=C['antithetic_mc'],lw=1.5,zorder=4,label='Permutation sampling')
ax[1].axvline(0,color='#64717A',lw=.8);lo=min(0,chosen.pccv_attribution.min()*1000);hi=chosen.pccv_attribution.max()*1000
ax[1].set(xlabel='Signed contribution (10⁻³ BC units)',xlim=(lo-.26,hi+1.5))
for yi,(_,row) in zip(y,chosen.iterrows()):
    value=row.pccv_attribution*1000
    ax[1].text(max(value,0)+.12,yi,f'{value:.3f}  (#{int(row.pccv_rank)})',va='center',fontsize=13)
for a in ax:a.grid(axis='x',color='#E9EDF0',lw=.6);a.set_axisbelow(True);a.tick_params(axis='y',length=0)
panel(ax[0],'A','Observed abundance and rank');panel(ax[1],'B','Model contribution and rank')
fig.legend(*ax[1].get_legend_handles_labels(),loc='outside upper right',ncol=2,frameon=False,fontsize=13)
fig.suptitle(f'{sample}  ·  {pathway}  ·  78 carriers',fontsize=15,y=1.09)
save(fig,'Figure10_real_carrier_case',chosen,['results/real_attributions.csv','results/real_case_plan.json'])

# Editable vector framework, with operations taken from the manuscript.
fig,ax=plt.subplots(figsize=(10.8,7.4));fig.subplots_adjust(0,0,1,1)
ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
def box(x,y,w,h,face='#F4F7F9',edge='#B9C8D2',lw=1.0):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.009,rounding_size=0.008',facecolor=face,edgecolor=edge,linewidth=lw);ax.add_patch(p)
def text(x,y,s,size=14,weight='normal',ha='center',va='center',color='#20262D'):
    return ax.text(x,y,s,fontsize=size,weight=weight,ha=ha,va=va,color=color,linespacing=1.25)
def arrow(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=14,lw=1.15,color='#526675'))
text(.035,.968,'A',18,'bold',ha='left');text(.08,.968,'Carrier attribution workflow',17,'bold',ha='left')
xs=[.035,.228,.421,.614,.807];w=.157;y0=.646;h=.257
headings=['Frozen inputs','Composition\nmodel','Carrier deletion\ngame','Shapley\nestimation','Carrier\nhypotheses']
bodies=['Species presence P\nFixed annotations Z\n$X_{ik}=P_i Z_{ik}$',
 'Shared projection\nTwo attention blocks\nAll-species softmax',
 'Select community, k\nAblate coordinate k\nOther inputs fixed',
 'Permutation sampling\nor PCCV\nSame attribution target',
 'Signed contributions\nCarrier rankings\nTargets for follow-up']
for x,title,body in zip(xs,headings,bodies):
    box(x,y0,w,h,face='#F0F5F8');text(x+w/2,.858,title,14.1,'bold');text(x+w/2,.753,body,13)
for a,b in zip(xs[:-1],xs[1:]):arrow(a+w+.006,.77,b-.009,.77)
text(.5,.606,r'$v(S)=\mathrm{BC}(f(X),f(X^{S,0})),\qquad \Sigma_i\varphi_i=v(C)$',16)
box(.035,.205,.303,.325,face='#F7F9FA')
text(.052,.5,'B',17,'bold',ha='left');text(.081,.5,'Exact repeated-token inference',14.5,'bold',ha='left')
text(.186,.447,'Identical absent-species inputs',13)
for i in range(4):box(.078+i*.052,.387,.03,.033,face='#C9D5DC',edge='#A6BAC6',lw=.7)
arrow(.186,.371,.186,.345);box(.132,.295,.108,.043,face='#DBE6ED',edge='#A6BAC6',lw=.7)
text(.186,.316,'One token × n',13)
text(.186,.247,'Attention logits + log n\nFinal softmax multiplicity correction',13)
box(.383,.205,.581,.325,face='#F0F5F8',edge='#94B1C7')
text(.4,.5,'C',17,'bold',ha='left');text(.43,.5,'PCCV at a matched query budget',14.5,'bold',ha='left')
cx=[.402,.545,.688,.831];cw=.116
for x,title,detail in zip(cx,['Probe','Fit','Sample','Correct'],['Endpoints +\neight masks','Cubic proxy\nAnalytical φ(h)','Reverse-paired\npermutations','Cross-fitted\nresiduals']):
    box(x,.351,cw,.093,face='white',edge='#A7BDCD',lw=.8)
    text(x+cw/2,.421,title,13,'bold');text(x+cw/2,.383,detail,13)
for a,b in zip(cx[:-1],cx[1:]):arrow(a+cw+.004,.395,b-.008,.395)
text(.675,.301,r'$h(z)=a^{T}z+b(w^{T}z)^2+c(w^{T}z)^3$',15)
text(.675,.25,'Calibration probes counted in the query budget',12.7)
box(.035,.036,.929,.108,face='white',edge='#B9C8D2')
text(.052,.112,'D',17,'bold',ha='left');text(.086,.112,'Verification resources',14.5,'bold',ha='left')
text(.502,.073,'90 controlled games   |   108 real cases   |   Six exact real conditional subgames',14)
save(fig,'Figure9_method_framework',source_names=['docs/METHODS.md','src/cv_shapley.py','src/common.py'])
quality={'font':'Times New Roman','font_path':font_manager.findfont('Times New Roman',fallback_to_default=False),
 'palette':C,'method_labels':N,'figures':records,'frozen_source_sha256':sources,
 'new_experiments':False,'case_carriers':len(out),'case_negative_carriers':meta['negative_pccv_carriers']}
(ROOT/'figure_style_and_sources.json').write_text(json.dumps(quality,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'figures':len(records),'font':quality['font'],'case':meta,'frozen_results_modified':False},ensure_ascii=False))
