from pathlib import Path
from statistics import median
import csv,hashlib,json,math,xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent;FROZEN=ROOT.parent
def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-13,abs_tol=1e-16),(a,b)
style=json.loads((ROOT/'figure_style_and_sources.json').read_text(encoding='utf8'))
for rel,digest in style['frozen_source_sha256'].items():
    assert hashlib.sha256((FROZEN/rel).read_bytes()).hexdigest()==digest
assert len(style['figures'])==10 and style['font']=='Times New Roman'
ns={'s':'http://www.w3.org/2000/svg'};font_checks=[]
for record in style['figures']:
    path=ROOT/'figures'/(record['figure']+'.svg');tree=ET.parse(path)
    labels=tree.findall('.//s:text',ns)
    for text in tree.iter():
        css=text.attrib.get('style','')
        if 'font-family:' in css:assert 'Times New Roman' in css,(path.name,css)
    assert '\u25a1' not in ''.join(tree.getroot().itertext())
    font_checks.append({'figure':record['figure'],'svg_text_elements':len(labels),'font_pass':True})

attrs=rows(FROZEN/'results/real_attributions.csv')
plan=json.loads((FROZEN/'results/real_case_plan.json').read_text(encoding='utf8'))
case=rows(ROOT/'case_study/Case_Study_S1.csv')
sample=plan['test_sample_ids'][0];pathway=plan['pathways'][0]['id']
original=[x for x in attrs if x['sample_id']==sample and x['pathway_id']==pathway]
by_method={method:{x['species_id']:x for x in original if x['method']==method}
           for method in ['probe_cubic_cv','antithetic_mc','proportional']}
assert len(case)==78 and len({x['species_id'] for x in case})==78
rank_maps={}
for method,prefix in [('probe_cubic_cv','pccv'),('antithetic_mc','permutation'),('proportional','allocation')]:
    ordered=sorted(by_method[method].values(),key=lambda x:(-float(x['attribution']),x['species_id']))
    rank_maps[prefix]={row['species_id']:i+1 for i,row in enumerate(ordered)}
abundance_order=sorted(by_method['probe_cubic_cv'].values(),key=lambda x:(-float(x['observed_abundance']),x['species_id']))
rank_maps['abundance']={row['species_id']:i+1 for i,row in enumerate(abundance_order)}
for row in case:
    sp=row['species_id'];assert row['sample_id']==sample and row['pathway_id']==pathway
    close(row['observed_abundance'],by_method['probe_cubic_cv'][sp]['observed_abundance'])
    close(row['annotation_score'],by_method['probe_cubic_cv'][sp]['annotation_score'])
    close(row['relative_abundance_percent'],100*float(row['observed_abundance']))
    for method,prefix in [('probe_cubic_cv','pccv'),('antithetic_mc','permutation'),('proportional','allocation')]:
        close(row[prefix+'_attribution'],by_method[method][sp]['attribution'])
        assert int(row[prefix+'_rank'])==rank_maps[prefix][sp]
    assert int(row['abundance_rank'])==rank_maps['abundance'][sp]
ordered=sorted(case,key=lambda x:int(x['pccv_rank']))
selected={x['species_id'] for x in abundance_order[:3]}|{x['species_id'] for x in ordered[:3]}|{ordered[-1]['species_id']}
assert selected=={x['species_id'] for x in case if x['displayed_in_figure']=='True'} and len(selected)==6
assert sum(float(x['pccv_attribution'])<0 for x in case)==9
assert rank_maps['abundance']['Ruminococcus_gnavus']==1 and rank_maps['pccv']['Ruminococcus_gnavus']==53
assert rank_maps['permutation']['Ruminococcus_gnavus']==51
assert rank_maps['abundance']['Butyrivibrio_sp_CAG_318']==44 and rank_maps['pccv']['Butyrivibrio_sp_CAG_318']==2

checked_medians=0
for filename,frozen_files in [
 ('Figure3_matched_budget_estimation.csv',['results/cv_matched_budget_evaluation.csv']),
 ('Figure8_larger_games_and_kernel_baseline.csv',['results/larger_game_matched_budget.csv','results/poly_shap_matched_budget.csv'])]:
    raw=[row for file in frozen_files for row in rows(FROZEN/file)]
    for row in rows(ROOT/'figure_data'/filename):
        values=[float(x['normalized_l1_error']) for x in raw if x['family']==row['family'] and x['method']==row['method'] and x['budget_permutations']==row['budget_permutations']]
        close(row['median_error'],median(values));checked_medians+=1
controlled=rows(FROZEN/'results/controlled_attribution_metrics.csv')
for row in rows(ROOT/'figure_data/Figure2_ground_truth_attribution.csv'):
    if row['panel']=='A':
        values=[float(x['normalized_l1_error']) for x in controlled if x['family']==row['family'] and x['method']==row['method']]
        close(row['median_error'],median(values))
    else:
        values=[float(x['dummy_mass_fraction']) for x in controlled if x['family']!='null' and x['method']==row['method']]
        assert len(values)==40
        close(row['null_mass_fraction'],median(values))
    checked_medians+=1
curves=rows(FROZEN/'results/real_deletion_curves.csv');checked_curve_points=0
for row in rows(ROOT/'figure_data/Figure5_real_perturbation_faithfulness.csv'):
    raw=[x for x in curves if x['method']==row['method'] and float(x['fraction'])==float(row['fraction'])]
    assert len(raw)==108
    close(row['deletion_normalized'],sum(float(x['deletion_bc'])/float(x['joint_bc']) for x in raw)/len(raw))
    close(row['restoration_normalized'],sum(float(x['restoration_gain_bc'])/float(x['joint_bc']) for x in raw)/len(raw))
    checked_curve_points+=2
exact=rows(FROZEN/'results/real_exact_subgames.csv');checked_exact_medians=0
for row in rows(ROOT/'figure_data/Figure6_exactness_and_sampling.csv'):
    values=[float(x['normalized_l1_error']) for x in exact if x['method']==row['method'] and x['case_id']==row['case_id']]
    close(row['median_error'],median(values));checked_exact_medians+=1
original_compression=rows(FROZEN/'results/compression_benchmark.csv')
display_compression=rows(ROOT/'figure_data/Figure1_exact_sparse_inference.csv')
assert len(original_compression)==len(display_compression)==18
for a,b in zip(original_compression,display_compression):
    assert a['sample_id']==b['sample_id']
    for key in a:
        if key!='sample_id':close(a[key],b[key])
profile=rows(ROOT/'figure_data/Figure7_carrier_context_profiles.csv')
profile_raw=[x for x in attrs if x['method']=='probe_cubic_cv' and x['case_id'].endswith('p00')]
communities=sorted({x['case_id'] for x in profile_raw});species={x['species_id'] for x in profile_raw}
lookup={(x['species_id'],x['case_id']):float(x['attribution']) for x in profile_raw}
top=sorted(species,key=lambda sp:(-sum(abs(lookup.get((sp,c),0)) for c in communities)/len(communities),sp))[:14]
assert [x['species_id'] for x in profile]==top
for row in profile:
    for c in communities:
        denominator=sum(abs(lookup.get((sp,c),0)) for sp in top)+1e-12
        close(row[c],lookup.get((row['species_id'],c),0)/denominator)
result={'all_pass':True,'figure_count':10,'main_figures':9,'supplementary_figures':1,
        'font':'Times New Roman','font_checks':font_checks,'matched_budget_and_control_medians_verified':checked_medians,
        'normalized_curve_values_verified':checked_curve_points,'exact_conditional_medians_verified':checked_exact_medians,
        'compression_records_verified':18,'profile_display_cells_verified':len(profile)*len(communities),'case_complete_carriers_verified':78,
        'displayed_carriers':6,'negative_carriers':9,'selection':'First frozen sample and pathway; display selection rule disclosed',
        'source_values_and_ranks_match':True,
        'summary_correction':{'panel':'Figure 4B','reason':'Preserve the literal null family label instead of parsing it as missing data; explicitly exclude ten fully null games',
                              'nonnull_games_per_method':40,'annotation_allocation_null_mass_median':median(float(x['dummy_mass_fraction']) for x in controlled if x['family']!='null' and x['method']=='proportional')},
        'new_experiments':False,'frozen_results_modified':False}
(ROOT/'figure_revision_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k!='font_checks'},ensure_ascii=False))
