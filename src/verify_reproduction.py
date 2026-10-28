from cv_shapley import *
from scipy.stats import spearmanr
import pandas as pd
sp,sa,pa,A,F,_=load_data();checks=[]
spl=pd.read_csv(ROOT/'results/sample_split.csv');test=[sa.index(s) for s in spl[spl.split=='test'].sample_id]
def jsd(p,q):
    mid=(p+q)/2;return .5*((p*np.log((p+1e-30)/(mid+1e-30))).sum(-1)+(np.where(q>0,q*np.log((q+1e-30)/(mid+1e-30)),0)).sum(-1))
for rep in ['annotation','fixed_shuffled_annotation']:
    Z=np.load(ROOT/'results'/f'{rep}_features_scaler.npz')['Z']
    for seed in [11,22,33]:
        net=Frozen();net.load_state_dict(torch.load(ROOT/'results'/f'{rep}_seed{seed}.pth',map_location='cpu',weights_only=True));net.eval()
        original=pd.read_csv(ROOT/'results'/f'{rep}_seed{seed}_test.csv').set_index('sample_id');diff=0
        for g in test:
            ids,x,c=compressed_input(g,A,Z);p=forward_batches(net,x[None],c)[0];y=np.r_[A[ids,g],0]
            diff=max(diff,abs(float(bc(p,y))-original.loc[sa[g],'bc']),abs(float(jsd(p,y))-original.loc[sa[g],'jsd']))
        checks.append({'check':f'{rep}_seed{seed}_test_reload','max_error':diff,'tolerance':1e-5,'pass':diff<1e-5})
# The independently sourced study holdout is verified from its own fitted scaler.
holdroot=ROOT/'study_holdout/results';hd=pd.read_csv(holdroot/'sample_split.csv');holdtest=[sa.index(s) for s in hd[hd.split=='test'].sample_id]
hz=np.load(holdroot/'annotation_features_scaler.npz')['Z'];hn=Frozen();hn.load_state_dict(torch.load(holdroot/'annotation_seed11.pth',map_location='cpu',weights_only=True));hn.eval()
hr=pd.read_csv(holdroot/'annotation_seed11_test.csv').set_index('sample_id');diff=0
for g in holdtest:
    ids,x,c=compressed_input(g,A,hz);p=forward_batches(hn,x[None],c)[0];y=np.r_[A[ids,g],0]
    diff=max(diff,abs(float(bc(p,y))-hr.loc[sa[g],'bc']),abs(float(jsd(p,y))-hr.loc[sa[g],'jsd']))
checks.append({'check':'study_holdout_seed11_test_reload','max_error':diff,'tolerance':1e-5,'pass':diff<1e-5})
Z=np.load(ROOT/'results/annotation_features_scaler.npz')['Z'];net=Frozen();net.load_state_dict(torch.load(ROOT/'results/annotation_seed11.pth',map_location='cpu',weights_only=True));net.eval()
for p in net.parameters():p.requires_grad_(False)
plan=json.loads((ROOT/'results/real_case_plan.json').read_text());k=pa.index(plan['pathways'][0]['id'])
for si in [0,6,12]:
    g=sa.index(plan['test_sample_ids'][si]);ids,x,c=compressed_input(g,A,Z);players=np.flatnonzero(F[ids,k]>0);game=DeletionGame(net,x,c,k,players);stored=np.load(ROOT/'results/real_cases'/f's{si:02d}_p00.npz')
    a,_=mc_shapley(game,len(players),32,20261028+si*100);b,_,_=cv_shapley(game,len(players),32,20261028+si*100)
    diff=max(float(np.max(np.abs(a-stored['mc1']))),float(np.max(np.abs(b-stored['cv1']))))
    checks.append({'check':f'real_case_s{si:02d}_independent_rerun','max_error':diff,'tolerance':1e-6,'pass':diff<1e-6});print(checks[-1],flush=True)
# Reload every fitted controlled predictor and independently recompute its outputs.
for file in sorted((ROOT/'results').glob('controlled_*_seed*.npz')):
    d=np.load(file);model=torch.nn.Sequential(torch.nn.Linear(8,32),torch.nn.Tanh(),torch.nn.Linear(32,12),torch.nn.Softmax(dim=1));model.load_state_dict(torch.load(file.with_suffix('.pth'),map_location='cpu',weights_only=True));model.eval()
    with torch.no_grad():p=model(torch.tensor(d['masks'])).numpy()
    diff=float(np.max(np.abs(p-d['predictions'])));checks.append({'check':file.stem+'_reload','max_error':diff,'tolerance':1e-7,'pass':diff<1e-7})
table('results/independent_reproduction_checks.csv',checks);dump('results/independent_reproduction_summary.json',{'checks':len(checks),'all_pass':all(c['pass'] for c in checks),'failed':[c for c in checks if not c['pass']]});assert all(c['pass'] for c in checks)
