from common import *
import math
cfg=json.loads((ROOT/'config.json').read_text());m=cfg['controlled_benchmark_players'];masks=enumerate_masks(m)
rows=[];fits=[];attrs=[];checks=[]
families=['heterogeneous','redundant','cooperative','compensatory','null']
for family in families:
    for seed in cfg['controlled_benchmark_seeds']:
        rng=np.random.default_rng(seed);base=rng.normal(0,.4,12);B=rng.normal(0,.35,(m,12));B[4:]=0
        if family=='redundant':B[:4]=B[0]
        if family=='compensatory':B[1]=-B[0];B[3]=-B[2]
        if family=='null':B[:]=0
        def oracle(mask):
            mm=np.asarray(mask);logits=base+mm@B
            if family=='cooperative':logits=logits+(mm[:,0]*mm[:,1])[:,None]*B[0]*3+(mm[:,2]*mm[:,3])[:,None]*B[2]*2
            pp=np.exp(logits-logits.max(1,keepdims=True));return pp/pp.sum(1,keepdims=True)
        truep=oracle(masks);p0=oracle(np.zeros((1,m)))[0];vv=bc(truep,p0);truth=exact_shapley(vv,m)
        if family=='null':assert np.max(np.abs(vv))<1e-12
        assert np.max(np.abs(truth[4:]))<1e-12
        denom=max(np.abs(truth).sum(),1e-12)
        abundance=rng.lognormal(0,1,m);annotation=rng.lognormal(0,.5,m);total=vv[-1]
        lookup=lambda mm:vv[(np.asarray(mm).astype(int)*(1<<np.arange(m))).sum(1)]
        end,loo,_=endpoint(lookup,m)
        methods={'proportional':total*abundance*annotation/(abundance*annotation).sum(),'abundance':total*abundance/abundance.sum(),'uniform':np.repeat(total/m,m),'single_deletion':loo,'endpoint_average':end,'exact_shapley':truth}
        for nperm in [16,32,128]:methods[f'permutation_shapley_{nperm}']=mc_shapley(lookup,m,nperm,seed+700)[0]
        for method,a in methods.items():
            rows.append({'family':family,'seed':seed,'method':method,'normalized_l1_error':float(np.abs(a-truth).sum()/denom),'efficiency_abs_error':float(abs(a.sum()-total)),'dummy_allocation_abs_mass':float(np.abs(a[4:]).sum()),'dummy_mass_fraction':float(np.abs(a[4:]).sum()/max(np.abs(a).sum(),1e-12)),'total_response':float(total),'truth_l1':float(np.abs(truth).sum())})
            for j,v in enumerate(a):attrs.append({'family':family,'seed':seed,'method':method,'player':j,'true_dummy':int(j>=4),'attribution':float(v),'exact_truth':float(truth[j])})
        checks.append({'family':family,'seed':seed,'exact_efficiency_error':float(abs(truth.sum()-total)),'max_dummy_attribution':float(np.max(np.abs(truth[4:])))})
        if family=='null':continue
        # Fit an independently parameterized predictor to a strict coalition split.
        # The observed input here is intervention status, which is available in
        # this controlled benchmark, not in the observational human dataset.
        torch.manual_seed(seed+9100);net=torch.nn.Sequential(torch.nn.Linear(m,32),torch.nn.Tanh(),torch.nn.Linear(32,12),torch.nn.Softmax(dim=1))
        ix=rng.permutation(np.arange(1,2**m-1));tr=np.r_[0,2**m-1,ix[:152]];va=ix[152:203];te=ix[203:]
        xx=torch.tensor(masks);yy=torch.tensor(truep,dtype=torch.float32);opt=torch.optim.Adam(net.parameters(),lr=.015)
        best=1e9;state=None;stale=0
        for ep in range(800):
            opt.zero_grad(set_to_none=True);pp=net(xx[tr]);loss=(pp-yy[tr]).square().mean();loss.backward();opt.step()
            with torch.no_grad():val=float((net(xx[va])-yy[va]).square().mean())
            if val<best-1e-9:best=val;state={k:v.detach().clone() for k,v in net.state_dict().items()};stale=0
            else:stale+=1
            if stale>=60:break
        net.load_state_dict(state);net.eval()
        with torch.no_grad():pred=net(xx).numpy()
        learnedv=bc(pred,pred[0]);learnedphi=exact_shapley(learnedv,m)
        fits.append({'family':family,'seed':seed,'epochs':ep+1,'n_train_coalitions':len(tr),'n_validation_coalitions':len(va),'n_test_coalitions':len(te),'test_bc':float(bc(pred[te],truep[te]).mean()),'exact_model_vs_mechanism_normalized_l1':float(np.abs(learnedphi-truth).sum()/denom),'learned_dummy_mass_fraction':float(np.abs(learnedphi[4:]).sum()/max(np.abs(learnedphi).sum(),1e-12))})
        np.savez_compressed(ROOT/'results'/f'controlled_{family}_seed{seed}.npz',B=B,base=base,masks=masks,truth_predictions=truep,predictions=pred,train=tr,validation=va,test=te,exact_truth=truth,learned_exact=learnedphi)
        torch.save(net.state_dict(),ROOT/'results'/f'controlled_{family}_seed{seed}.pth')
        print(family,seed,'test_bc',fits[-1]['test_bc'],flush=True)
table('results/controlled_attribution_metrics.csv',rows);table('results/controlled_attributions.csv',attrs);table('results/controlled_predictor_fits.csv',fits);table('results/controlled_exact_checks.csv',checks)
summ=[]
for family in families:
    for method in sorted({r['method'] for r in rows}):
        sub=[r for r in rows if r['family']==family and r['method']==method]
        summ.append({'family':family,'method':method,'n_seeds':len(sub),'median_normalized_l1_error':float(np.median([r['normalized_l1_error'] for r in sub])),'median_dummy_mass_fraction':float(np.median([r['dummy_mass_fraction'] for r in sub])),'max_efficiency_abs_error':max(r['efficiency_abs_error'] for r in sub)})
table('results/controlled_summary.csv',summ)
dump('results/controlled_verification.json',{'families':families,'cases':len(checks),'fitted_predictors':len(fits),'max_exact_efficiency_error':max(v['exact_efficiency_error'] for v in checks),'max_exact_dummy_attribution':max(v['max_dummy_attribution'] for v in checks),'all_null_cases_zero':all(r['total_response']<1e-12 for r in rows if r['family']=='null')})
