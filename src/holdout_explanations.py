from cv_shapley import *
import csv
sp,sa,pa,A,F,_=load_data();Z=np.load(ROOT/'study_holdout/results/annotation_features_scaler.npz')['Z'];net=Frozen();net.load_state_dict(torch.load(ROOT/'study_holdout/results/annotation_seed11.pth',map_location='cpu',weights_only=True));net.eval()
for p in net.parameters():p.requires_grad_(False)
with (ROOT/'study_holdout/results/sample_split.csv').open(encoding='utf-8-sig',newline='') as f:split={r['sample_id']:r['split'] for r in csv.DictReader(f)}
chosen=sorted([s for s in sa if split[s]=='test'],key=lambda s:hashlib.sha256(s.encode()).hexdigest())[:3]
plan=json.loads((ROOT/'results/real_case_plan.json').read_text());rows=[]
for sample in chosen:
    g=sa.index(sample);ids,x,c=compressed_input(g,A,Z)
    for pdesc in plan['pathways']:
        k=pa.index(pdesc['id']);players=np.flatnonzero(F[ids,k]>0);m=len(players)
        if m<2:continue
        game=DeletionGame(net,x,c,k,players);phi,_,_=cv_shapley(game,m,32,4419+g+k);_,_,total=endpoint(game,m);weights=A[ids[players],g]*F[ids[players],k];weights/=weights.sum()
        for method,val in [('proportional',total*weights),('probe_cubic_cv',phi)]:
            fractions=np.array([0,.1,.2,.4,.6,.8,1]);order=np.argsort(-val);mm=np.zeros((len(fractions),m))
            for j,fraction in enumerate(fractions):mm[j,order[:int(np.ceil(fraction*m))]]=1
            dele=game(mm);ins=total-game(1-mm)
            rows.append({'sample_id':sample,'pathway_id':pa[k],'method':method,'joint_bc':float(total),'deletion_auc':float(np.trapezoid(dele,fractions)/max(float(total),1e-9)),'restoration_auc':float(np.trapezoid(ins,fractions)/max(float(total),1e-9)),'efficiency_error':float(abs(val.sum()-total))})
        print(sample,pa[k],'holdout audit complete',flush=True)
        table('results/study_holdout_explanation_audit.csv',rows)
