from common import *
import argparse,copy
cfg=json.loads((ROOT/'config.json').read_text())
parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');parser.add_argument('--study-holdout',action='store_true');parser.add_argument('--force',action='store_true');parser.add_argument('--replay',action='store_true');args=parser.parse_args()
BASE=ROOT
if args.study_holdout or args.replay:
    import common as C
    ROOT=ROOT/('study_holdout' if args.study_holdout else 'replay');C.ROOT=ROOT;(ROOT/'results').mkdir(parents=True,exist_ok=True)
sp,sa,pa,A,F,_=load_data()
rng=np.random.default_rng(cfg['split_seed']);order=rng.permutation(len(sa));ntr=int(len(sa)*.7);nv=int(len(sa)*.15)
tr,va,te=order[:ntr],order[ntr:ntr+nv],order[ntr+nv:]
if args.study_holdout:
    import collections
    with (BASE/'results/sample_metadata_verified.csv').open(encoding='utf-8-sig',newline='') as f:metadata={r['sample_id']:r for r in csv.DictReader(f)}
    usable=[i for i,s in enumerate(sa) if metadata[s]['unambiguous_study_subject']=='1'];counts=collections.Counter(metadata[sa[i]]['study_name'] for i in usable);heldout=counts.most_common(1)[0][0]
    te=np.array([i for i in usable if metadata[sa[i]]['study_name']==heldout]);groups=sorted({metadata[sa[i]]['subject_key'] for i in usable if metadata[sa[i]]['study_name']!=heldout})
    rng.shuffle(groups);valid=set(groups[:max(1,int(len(groups)*.15))]);va=np.array([i for i in usable if metadata[sa[i]]['study_name']!=heldout and metadata[sa[i]]['subject_key'] in valid]);tr=np.array([i for i in usable if metadata[sa[i]]['study_name']!=heldout and metadata[sa[i]]['subject_key'] not in valid])
    dump('results/holdout_design.json',{'heldout_study':heldout,'selection':'largest study by verified sample count, chosen without examining model performance','n_train':len(tr),'n_validation':len(va),'n_test':len(te),'n_excluded_unverified_or_ambiguous':len(sa)-len(usable),'train_validation_subject_overlap':len({metadata[sa[g]]['subject_key'] for g in tr}&{metadata[sa[g]]['subject_key'] for g in va})})
split={sa[i]:('train' if i in set(tr) else 'validation' if i in set(va) else 'test' if i in set(te) else 'excluded_unverified_metadata') for i in range(len(sa))}
table('results/sample_split.csv',[{'sample_id':s,'split':split[s]} for s in sa])
assert not (set(tr)&set(te) or set(tr)&set(va) or set(va)&set(te))
P=A>0;ids=[np.flatnonzero(P[:,g]) for g in range(len(sa))]
Y=[np.r_[A[ii,g],0.0].astype('float32') for g,ii in enumerate(ids)]
prior=A[:,tr].mean(1)+1e-10
def jsd(p,q):
    m=(p+q)/2
    return .5*((np.where(p>0,p*np.log((p+1e-30)/(m+1e-30)),0)).sum(-1)+(np.where(q>0,q*np.log((q+1e-30)/(m+1e-30)),0)).sum(-1))
def scaler(FF):
    weights=P[:,tr].sum(1);V=FF>0;LL=np.log1p(FF);cnt=(weights[:,None]*V).sum(0)
    mu=(weights[:,None]*LL).sum(0)/np.maximum(cnt,1)
    sd=np.sqrt((weights[:,None]*(LL-mu)**2*V).sum(0)/np.maximum(cnt,1));sd[sd<1e-12]=1
    return np.where(V,(LL-mu)/sd,0).astype('float32'),mu,sd,cnt
def batch(gids,Z):
    length=max(len(ids[g])+1 for g in gids);b=len(gids)
    xx=np.zeros((b,length,463),dtype='float32');yy=np.zeros((b,length),dtype='float32');cc=np.zeros((b,length),dtype='float32')
    for j,g in enumerate(gids):
        n=len(ids[g]);xx[j,:n]=Z[ids[g]];yy[j,:n+1]=Y[g];cc[j,:n]=1;cc[j,n]=1090-n
    return torch.from_numpy(xx),torch.from_numpy(yy),torch.from_numpy(cc)
@torch.inference_mode()
def evaluate(net,Z,idx):
    results=[]
    for st in range(0,len(idx),16):
        gg=idx[st:st+16];x,y,c=batch(gg,Z);p=net(x,c).numpy();q=y.numpy()
        for j,g in enumerate(gg):results.append({'sample_id':sa[g],'bc':float(bc(p[j],q[j])),'jsd':float(jsd(p[j],q[j])),'absent_mass':float(p[j,len(ids[g])])})
    return results
prior_rows=[]
for g in te:
    pp=prior[ids[g]];pp=pp/pp.sum();qq=A[ids[g],g];qq=qq/qq.sum()
    prior_rows.append({'sample_id':sa[g],'bc':float(bc(pp,qq)),'jsd':float(jsd(pp,qq)),'absent_mass':0.0})
table('results/presence_conditioned_prior_test.csv',prior_rows)
representations=cfg['representations'][:1] if args.pilot or args.study_holdout or args.replay else cfg['representations']
seeds=cfg['training_seeds'][:1] if args.pilot or args.study_holdout or args.replay else cfg['training_seeds'];summary=[]
for rep in representations:
    FF=F.copy()
    if rep!='annotation':FF=FF[np.random.default_rng(1729).permutation(len(F))]
    Z,mu,sd,cnt=scaler(FF);np.savez_compressed(ROOT/'results'/f'{rep}_features_scaler.npz',Z=Z,mu=mu,sigma=sd,counts=cnt)
    for seed in seeds:
        dest=ROOT/'results'/f'{rep}_seed{seed}.pth';meta=dest.with_suffix('.json')
        if dest.exists() and meta.exists() and not args.pilot and not args.force:
            summary.append(json.loads(meta.read_text()));print('resume',rep,seed,flush=True);continue
        torch.manual_seed(seed);net=Frozen();net.train();opt=torch.optim.AdamW(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        hist=[];best=float('inf');best_state=None;stale=0;timer=time.perf_counter()
        for epoch in range(3 if args.pilot else cfg['max_epochs']):
            net.train();rr=np.random.default_rng(seed*1000+epoch).permutation(tr);losses=[]
            for st in range(0,len(rr),cfg['batch_size']):
                x,y,c=batch(rr[st:st+cfg['batch_size']],Z);opt.zero_grad(set_to_none=True);p=net(x,c)
                loss=torch.mean(torch.abs(p-y).sum(1)/torch.abs(p+y).sum(1).clamp_min(1e-10));loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.0);opt.step();losses.append(float(loss.detach()))
            net.eval();vr=evaluate(net,Z,va);v=float(np.mean([z['bc'] for z in vr]));hist.append({'epoch':epoch+1,'training_bc':float(np.mean(losses)),'validation_bc':v,'elapsed_seconds':time.perf_counter()-timer})
            print(rep,seed,hist[-1],flush=True)
            if v<best-1e-5:best=v;best_state=copy.deepcopy(net.state_dict());stale=0
            else:stale+=1
            if stale>=cfg['early_stopping_patience']:break
        net.load_state_dict(best_state);net.eval();test=evaluate(net,Z,te)
        table(f'results/{rep}_seed{seed}_history'+('_pilot.csv' if args.pilot else '.csv'),hist)
        if not args.pilot:
            torch.save(best_state,dest);table(f'results/{rep}_seed{seed}_test.csv',test)
            rr={'representation':rep,'seed':seed,'epochs':len(hist),'selected_epoch':min(hist,key=lambda z:z['validation_bc'])['epoch'],'validation_bc':best,'test_bc':float(np.mean([z['bc'] for z in test])),'test_jsd':float(np.mean([z['jsd'] for z in test])),'test_mean_absent_mass':float(np.mean([z['absent_mass'] for z in test])),'seconds':time.perf_counter()-timer,'checkpoint_sha256':sha(dest)}
            dump(str(meta.relative_to(ROOT)),rr);summary.append(rr)
        else:dump('results/training_pilot.json',{'epochs':hist,'estimated_6_models_40_epochs_seconds':(time.perf_counter()-timer)/3*6*40,'finite_training':all(np.isfinite(h['training_bc']) for h in hist)})
table('results/model_test_summary.csv',summary)
