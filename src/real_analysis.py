from cv_shapley import *
from scipy.stats import spearmanr
import argparse
cfg=json.loads((ROOT/'config.json').read_text());parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['discover','evaluate','multiseed'],required=True);parser.add_argument('--force',action='store_true');args=parser.parse_args()
sp,sa,pa,A,F,_=load_data();Z=np.load(ROOT/'results/annotation_features_scaler.npz')['Z'];net=Frozen();net.load_state_dict(torch.load(ROOT/'results/annotation_seed11.pth',map_location='cpu',weights_only=True));net.eval()
for p in net.parameters():p.requires_grad_(False)
with (ROOT/'results/sample_split.csv').open(encoding='utf-8-sig',newline='') as f:split={r['sample_id']:r['split'] for r in csv.DictReader(f)}
def choose_samples(label,n):
    ix=[g for g,s in enumerate(sa) if split[s]==label];ix.sort(key=lambda g:(A[:,g]>0).sum());out=[]
    for part in np.array_split(ix,3):
        out.extend(sorted(part,key=lambda g:hashlib.sha256(sa[g].encode()).hexdigest())[:n//3])
    return [int(g) for g in out]
def corr(x,y):
    if np.std(x)<1e-15 or np.std(y)<1e-15:return None
    return float(spearmanr(x,y).statistic)
case_dir=ROOT/'results'/'real_cases';case_dir.mkdir(exist_ok=True)
if args.stage=='discover':
    dev=choose_samples('validation',33);values=[]
    for g in dev:
        ids,x,c=compressed_input(g,A,Z);p0=forward_batches(net,x[None],c)[0];out=[]
        for st in range(0,len(pa),32):
            ks=np.arange(st,min(st+32,len(pa)));xx=x[None].repeat(len(ks),1,1)
            for j,k in enumerate(ks):xx[j,:,k]=0
            out.extend(bc(forward_batches(net,xx,c),p0))
        values.append(out)
    V=np.asarray(values);means=V.mean(0);order=np.argsort(-means);tr=[g for g,s in enumerate(sa) if split[s]=='train']
    binary=F>0;count=binary.sum(0);local=(A[:,tr]>0).T@binary.astype(float);U=A[:,tr].T@F
    D=np.c_[count,local.mean(0),(local>0).mean(0),U.mean(0)];scale=np.quantile(D,.75,axis=0)-np.quantile(D,.25,axis=0);scale[scale<=0]=1;D=(D-np.median(D,axis=0))/scale
    top=order[:3].tolist();controls=[]
    for k in top:
        available=[j for j in range(len(pa)) if j not in top+controls]
        controls.append(min(available,key=lambda j:float(np.sum((D[k]-D[j])**2))))
    kk=top+controls;gg=choose_samples('test',18)
    plan={'discovery_sample_ids':[sa[g] for g in dev],'test_sample_ids':[sa[g] for g in gg],'pathways':[{'id':pa[k],'kind':'high_response' if k in top else 'matched_background','discovery_mean_bc':float(means[k])} for k in kk],'selection_rule':'3 highest validation mean responses and 1 nearest training-carrier-background control per selected pathway; 18 hash-selected test samples stratified by richness','exclusion_rule':'fewer than two present carriers'}
    dump('results/real_case_plan.json',plan)
    table('results/discovery_pathways.csv',[{'pathway_id':pa[k],'validation_mean_bc':float(means[k]),'validation_median_bc':float(np.median(V[:,k])),'rank':int(np.flatnonzero(order==k)[0]+1),'global_carriers':int(count[k])} for k in range(len(pa))])
    np.savez_compressed(ROOT/'results/discovery_responses.npz',samples=np.array([sa[g] for g in dev]),pathways=np.array(pa),bc=V)
    print(plan,flush=True)
elif args.stage=='evaluate':
    plan=json.loads((ROOT/'results/real_case_plan.json').read_text());metric=[];attribution=[];curves=[];convergence=[];excluded=[];exactrows=[]
    for si,sample in enumerate(plan['test_sample_ids']):
        g=sa.index(sample);ids,x,c=compressed_input(g,A,Z)
        for ki,pdesc in enumerate(plan['pathways']):
            k=pa.index(pdesc['id']);players=np.flatnonzero(F[ids,k]>0);m=len(players);cid=f's{si:02d}_p{ki:02d}';saved=case_dir/(cid+'.npz');info=case_dir/(cid+'.json')
            if m<2:excluded.append({'sample_id':sample,'pathway_id':pa[k],'n_carriers':m,'reason':'fewer than 2 carriers'});continue
            game=DeletionGame(net,x,c,k,players);timer=time.perf_counter()
            end,loo,total=endpoint(game,m);weights=A[ids[players],g]*F[ids[players],k];weights/=weights.sum()
            if saved.exists() and not args.force:
                dd=np.load(saved);mc=[dd['mc1'],dd['mc2']];cv=[dd['cv1'],dd['cv2']];se=[dd['mc_se1'],dd['cv_se1']]
            else:
                mc=[];cv=[];se=[]
                for rep,seed in enumerate(cfg['shapley_seeds']):
                    aa,ss=mc_shapley(game,m,32,seed+si*100+ki);bb,tt,meta=cv_shapley(game,m,32,seed+si*100+ki)
                    mc.append(aa);cv.append(bb)
                    if rep==0:se=[ss,tt]
                np.savez_compressed(saved,species_ids=np.array([sp[ids[j]] for j in players]),mc1=mc[0],mc2=mc[1],cv1=cv[0],cv2=cv[1],mc_se1=se[0],cv_se1=se[1],endpoint=end,single=loo,proportional=total*weights)
            meanmc=np.mean(mc,0);meancv=np.mean(cv,0);den=max(np.abs(meanmc).sum(),1e-12)
            methods={'proportional':total*weights,'abundance':total*A[ids[players],g]/A[ids[players],g].sum(),'uniform':np.repeat(total/m,m),'single_deletion':loo,'endpoint_average':end,'antithetic_mc':meanmc,'probe_cubic_cv':meancv}
            convergence.append({'case_id':cid,'sample_id':sample,'pathway_id':pa[k],'n_carriers':m,'mc_replicate_nl1_difference':float(np.abs(mc[0]-mc[1]).sum()/den),'cv_replicate_nl1_difference':float(np.abs(cv[0]-cv[1]).sum()/max(np.abs(meancv).sum(),1e-12)),'mc_replicate_rho':corr(*mc),'cv_replicate_rho':corr(*cv)})
            fractions=np.array([0,.1,.2,.4,.6,.8,1.]);randommask=np.random.default_rng(5500+si*100+ki).integers(0,2,(64,m));truth_random=game(randommask)
            # All metrics use independently drawn coalition probes, not fitting probes.
            for method,val in methods.items():
                order=np.argsort(-val);masks=np.zeros((len(fractions),m))
                for j,fraction in enumerate(fractions):masks[j,order[:int(np.ceil(fraction*m))]]=1
                deletion=game(masks);restored=game(1-masks);insertion=total-restored
                norm=max(float(total),1e-9);auc=float(np.trapezoid(deletion,fractions)/norm);iauc=float(np.trapezoid(insertion,fractions)/norm)
                metric.append({'case_id':cid,'sample_id':sample,'pathway_id':pa[k],'pathway_kind':pdesc['kind'],'n_carriers':m,'method':method,'joint_bc':float(total),'efficiency_abs_error':float(abs(val.sum()-total)),'nl1_difference_vs_mc':float(np.abs(val-meanmc).sum()/den),'rank_rho_vs_mc':corr(val,meanmc),'relative_deletion_auc':auc,'relative_insertion_auc':iauc,'random_coalition_mse':float(np.mean((randommask@val-truth_random)**2)),'negative_attribution_fraction':float(np.mean(val<0))})
                for j,fraction in enumerate(fractions):curves.append({'case_id':cid,'method':method,'fraction':fraction,'deletion_bc':float(deletion[j]),'restoration_gain_bc':float(insertion[j]),'joint_bc':float(total)})
                for j,ss in enumerate(players):attribution.append({'case_id':cid,'sample_id':sample,'pathway_id':pa[k],'species_id':sp[ids[ss]],'method':method,'attribution':float(val[j]),'observed_abundance':float(A[ids[ss],g]),'annotation_score':float(F[ids[ss],k])})
            # Exact conditional subgame uses eight hash-selected carriers, leaving
            # all remaining carriers unchanged. It does not claim to enumerate
            # the full 100-player game.
            if si in [0,6,12] and ki in [0,3]:
                selected=sorted(players,key=lambda j:hashlib.sha256(sp[ids[j]].encode()).hexdigest())[:8];mm=len(selected);sub=DeletionGame(net,x,c,k,selected)
                vv=sub(enumerate_masks(mm));ph=exact_shapley(vv,mm);den2=max(np.abs(ph).sum(),1e-12);ww=A[ids[selected],g]*F[ids[selected],k];ww/=ww.sum()
                look=lambda xx:vv[(np.asarray(xx,dtype=int)*(1<<np.arange(mm))).sum(1)]
                for seed in range(20):
                    a,_=mc_shapley(look,mm,32,seed+800);b,_,meta=cv_shapley(look,mm,32,seed+800)
                    for method,val in [('antithetic_mc',a),('probe_cubic_cv',b),('proportional',vv[-1]*ww)]:exactrows.append({'case_id':cid,'seed':seed,'n_players':mm,'method':method,'normalized_l1_error':float(np.abs(val-ph).sum()/den2),'efficiency_abs_error':float(abs(val.sum()-vv[-1]))})
                np.savez_compressed(case_dir/(cid+'_exact.npz'),values=vv,exact_shapley=ph,species_ids=np.array([sp[ids[j]] for j in selected]))
            dump(str(info.relative_to(ROOT)),{'sample_id':sample,'pathway_id':pa[k],'n_carriers':m,'joint_bc':float(total),'query_evaluations_this_run':game.calls,'seconds_this_run':time.perf_counter()-timer,'checkpoint':'annotation_seed11.pth','case_file_sha256':sha(saved)})
            print(cid,sample,pa[k],m,'seconds',round(time.perf_counter()-timer,2),flush=True)
            # Save progress after every case.
            table('results/real_attribution_metrics.csv',metric);table('results/real_attributions.csv',attribution);table('results/real_deletion_curves.csv',curves);table('results/real_sampling_convergence.csv',convergence);table('results/real_exact_subgames.csv',exactrows);table('results/real_excluded_cases.csv',excluded)
    dump('results/real_evaluation_complete.json',{'cases':len(convergence),'excluded':len(excluded),'methods':list(methods),'exact_subgames':len(set(r['case_id'] for r in exactrows))})
elif args.stage=='multiseed':
    plan=json.loads((ROOT/'results/real_case_plan.json').read_text());rows=[]
    for seed in [11,22,33]:
        net.load_state_dict(torch.load(ROOT/'results'/f'annotation_seed{seed}.pth',map_location='cpu',weights_only=True));net.eval()
        for si,sample in enumerate(plan['test_sample_ids']):
            g=sa.index(sample);ids,x,c=compressed_input(g,A,Z)
            for ki,pdesc in enumerate(plan['pathways']):
                k=pa.index(pdesc['id']);players=np.flatnonzero(F[ids,k]>0)
                if len(players)<2:continue
                game=DeletionGame(net,x,c,k,players);end,loo,total=endpoint(game,len(players))
                for j,ss in enumerate(players):rows.append({'case_id':f's{si:02d}_p{ki:02d}','training_seed':seed,'species_id':sp[ids[ss]],'endpoint_attribution':float(end[j]),'single_deletion':float(loo[j]),'joint_bc':float(total)})
    table('results/real_cross_model_endpoints.csv',rows);print('multiseed rows',len(rows),flush=True)
