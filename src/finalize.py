"""Check the numerical pipeline and write an English execution summary."""
from report_common import *
import pandas as pd

R=ROOT/'results'
s=json.loads((R/'final_scientific_summary.json').read_text(encoding='utf8'))
rep=json.loads((R/'independent_reproduction_summary.json').read_text(encoding='utf8'))
core=json.loads((R/'core_verification.json').read_text(encoding='utf8'))
if sha(R/'annotation_seed11.pth')==sha(ROOT/'replay/results/annotation_seed11.pth'):
    replay_error=0.0
else:
    import torch
    original=torch.load(R/'annotation_seed11.pth',map_location='cpu',weights_only=True)
    replay=torch.load(ROOT/'replay/results/annotation_seed11.pth',map_location='cpu',weights_only=True)
    replay_error=max(float((original[key]-replay[key]).abs().max()) for key in original)
rt=pd.read_csv(R/'annotation_seed11_test.csv').set_index('sample_id')
rr=pd.read_csv(ROOT/'replay/results/annotation_seed11_test.csv').set_index('sample_id')
test_error=float(np.max(np.abs(rt.loc[rr.index,'bc']-rr.bc)))
kernel=pd.read_csv(R/'kernel_exact_small_game_checks.csv')
exact=json.loads((R/'controlled_verification.json').read_text(encoding='utf8'))
hold=pd.read_csv(ROOT/'study_holdout/results/annotation_seed11_test.csv')
poly=json.loads((R/'poly_baseline_verification.json').read_text(encoding='utf8'))
checks={'core_output_and_gradient_equivalence':core['pass'],
 'independent_reload_and_case_reproduction':rep['all_pass'],
 'independent_training_weights_max_error':replay_error,'independent_training_test_bc_max_error':test_error,
 'independent_training_pass':replay_error<1e-7 and test_error<1e-7,
 'controlled_exact_max_efficiency_error':exact['max_exact_efficiency_error'],
 'controlled_exact_null_carrier_error':exact['max_exact_dummy_attribution'],
 'kernel_exhaustive_max_error':float(kernel.exact_kernel_error.max()),
 'real_cases':s['real_cases'],'exact_real_subgames':s['real_exact_subgames'],'trained_main_models':len(s['models']),
 'controlled_fitted_models':len(list(R.glob('controlled_*_seed*.pth'))),
 'larger_exact_games':len(list(R.glob('larger_*_seed*.npz'))),
 'study_holdout_completed':len(hold)==s['study_holdout']['n_test']==544,
 'figures':len(list((ROOT/'figures').glob('Figure*.svg')))}
checks['poly_baseline_implementation_checks_pass']=poly['all_pass'] and poly['rows']==1600
checks['all_required_computational_checks_pass']=all([
 core['pass'],rep['all_pass'],checks['independent_training_pass'],
 checks['controlled_exact_max_efficiency_error']<1e-10,checks['controlled_exact_null_carrier_error']<1e-10,
 checks['kernel_exhaustive_max_error']<1e-7,checks['real_cases']==108,checks['exact_real_subgames']==6,
 checks['trained_main_models']==6,checks['controlled_fitted_models']==40,checks['larger_exact_games']==40,
 checks['study_holdout_completed'],checks['figures']==8,checks['poly_baseline_implementation_checks_pass']])
dump('results/final_verification.json',checks)
assert checks['all_required_computational_checks_pass'],checks
(ROOT/'EXECUTION_SUMMARY.md').write_text(
 '# Execution summary\n\nAll required computational checks passed.\n\n'
 +str(rep['checks'])+' model reload and complete real-case checks passed.\n\n'
 'The independent-training fields compare the separately archived same-seed replay with the reference model. '
 'Verify mode does not retrain either model. Execution timings depend on hardware.\n',encoding='utf8')
print(json.dumps(checks),flush=True)
