from cv_shapley import *
from scipy.stats import spearmanr
sp,sa,pa,A,F,_=load_data();Z=np.load(ROOT/'results/annotation_features_scaler.npz')['Z'];net=Frozen();net.load_state_dict(torch.load(ROOT/'results/annotation_seed11.pth',map_location='cpu',weights_only=True));net.eval()
for p in net.parameters():p.requires_grad_(False)
plan=json.loads((ROOT/'results/real_case_plan.json').read_text());rows=[]
class JSDGame(DeletionGame):
    def __call__(self,masks,return_predictions=False):
        pp=super().__call__(masks,True)
        if return_predictions:return pp
        q=self.p0;mid=(pp+q)/2
        return .5*(np.sum(pp*np.log((pp+1e-30)/(mid+1e-30)),1)+np.sum(q*np.log((q+1e-30)/(mid+1e-30)),1))
for si,sample in enumerate(plan['test_sample_ids']):
    g=sa.index(sample);ids,x,c=compressed_input(g,A,Z);k=pa.index(plan['pathways'][0]['id']);players=np.flatnonzero(F[ids,k]>0)
    if len(players)<2:continue
    stored=np.load(ROOT/'results/real_cases'/f's{si:02d}_p00.npz');reference=(stored['cv1']+stored['cv2'])/2
    for label,game in [('jsd',JSDGame(net,x,c,k,players)),('positive_feature_median_baseline',DeletionGame(net,x,c,k,players,baseline=float(np.median(Z[F[:,k]>0,k]))))]:
        a,_,meta=cv_shapley(game,len(players),32,62600+si);end,loo,total=endpoint(game,len(players))
        rows.append({'sample_id':sample,'pathway_id':pa[k],'check':label,'n_carriers':len(players),'joint_response':float(total),'rank_rho_vs_primary_bc_zero_baseline':float(spearmanr(a,reference).statistic),'efficiency_error':float(abs(a.sum()-total))})
    if si in [0,3,6,9,12,15]:
        original=net.output_layer.weight.detach().clone();perm=torch.randperm(64,generator=torch.Generator().manual_seed(20261027))
        with torch.no_grad():net.output_layer.weight.copy_(original[:,perm])
        altered=DeletionGame(net,x,c,k,players);aa,_,_=cv_shapley(altered,len(players),32,76700+si);altered_total=float(altered(np.ones((1,len(players))))[0])
        with torch.no_grad():net.output_layer.weight.copy_(original)
        rows.append({'sample_id':sample,'pathway_id':pa[k],'check':'output_parameter_permutation','n_carriers':len(players),'joint_response':altered_total,'rank_rho_vs_primary_bc_zero_baseline':float(spearmanr(aa,reference).statistic),'efficiency_error':float(abs(aa.sum()-altered_total))})
    table('results/real_sensitivity_checks.csv',rows);print(si,sample,'sensitivity complete',flush=True)
