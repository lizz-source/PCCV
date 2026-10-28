"""Query-calibrated cubic control variate for carrier deletion games.

Uses standard Shapley/control-variate principles. This specialized surrogate
and its empirical utility must be compared at matched query budgets.
"""
from common import *

def cv_shapley(game,m,budget_permutations,seed,degree=3):
    if not isinstance(m,(int,np.integer)) or m<3:
        raise ValueError('PCCV requires an integer carrier count of at least three.')
    if not isinstance(budget_permutations,(int,np.integer)) or budget_permutations<12 or budget_permutations%4:
        raise ValueError('The nominal budget must be an integer at least 12 and divisible by four.')
    if degree not in (2,3):
        raise ValueError('degree must be 2 for the ablation or 3 for PCCV.')
    nperm=max(8,budget_permutations-4)
    if nperm%2:nperm-=1
    probes=np.r_[np.eye(m),1-np.eye(m),np.ones((1,m)),np.random.default_rng(913).integers(0,2,(8,m))]
    pv=game(probes);single=pv[:m];last=pv[-9];complement=pv[m:2*m]
    w=single+np.abs(last-complement);w[w<1e-10]=0
    w=w/w.sum() if w.sum()>0 else np.zeros(m)
    tt=probes@w;D=np.c_[probes,tt**2,tt**3]
    if degree==2:D[:,-1]=0
    ridge=np.eye(m+2)*1e-8;ridge[-2:,-2:]*=100
    beta=np.linalg.lstsq(np.r_[D,ridge],np.r_[pv,np.zeros(m+2)],rcond=None)[0]
    def surrogate(mm):
        mm=np.asarray(mm);t=mm@w;return mm@beta[:m]+beta[-2]*t*t+beta[-1]*t*t*t
    W=w.sum();Q=(w*w).sum()
    phi_s=beta[:m]+beta[-2]*w*W+beta[-1]*(w*W*W+.5*w*Q-.5*w*w*W)
    rng=np.random.default_rng(seed);perms=[]
    for _ in range(nperm//2):
        pp=rng.permutation(m);perms.extend([pp,pp[::-1]])
    masks=np.zeros((nperm*(m+1),m),dtype='float32')
    for j,pp in enumerate(perms):
        for k in range(m):masks[j*(m+1)+k+1]=masks[j*(m+1)+k];masks[j*(m+1)+k+1,pp[k]]=1
    aa=game(masks).reshape(nperm,m+1);bb=surrogate(masks).reshape(nperm,m+1)
    A=np.zeros((nperm,m));B=np.zeros_like(A)
    for j,pp in enumerate(perms):A[j,pp]=np.diff(aa[j]);B[j,pp]=np.diff(bb[j])
    A=A.reshape(-1,2,m).mean(1);B=B.reshape(-1,2,m).mean(1)
    halves=np.array_split(np.arange(len(A)),2);adjusted=np.zeros_like(A);coeff=[]
    for train,test in [(halves[0],halves[1]),(halves[1],halves[0])]:
        ac=A[train]-A[train].mean(0);bcv=B[train]-B[train].mean(0)
        c=float(np.clip(np.sum(ac*bcv)/(np.sum(bcv**2)+1e-30),-2,2));coeff.append(c)
        adjusted[test]=A[test]-c*(B[test]-phi_s)
    return adjusted.mean(0),adjusted.std(0,ddof=1)/np.sqrt(len(adjusted)),{'crossfit_coefficients':coeff,'theoretical_query_budget':nperm*(m+1)+2*m+9,'comparison_mc_budget':budget_permutations*(m+1)}
