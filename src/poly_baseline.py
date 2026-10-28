"""Independent implementation of Fumagalli et al. (ICLR 2026), Algorithm 1.

Full k-additive frontier; kernel-proportional sampling, with/without pairing.
This is not the authors' complete software or leverage-score implementation.
Source: https://arxiv.org/html/2601.18608v2, Equations 3--7.
"""
from common import *
from itertools import combinations
import math

def poly_shapley(game,m,budget,seed,degree=3,pairing=True):
    terms=[t for size in range(1,degree+1) for t in combinations(range(m),size)]
    if budget<=len(terms)+2:raise ValueError('Budget must exceed frontier size plus endpoints.')
    ends=np.asarray(game(np.stack([np.zeros(m),np.ones(m)])),dtype=float);v0=ends[0];total=ends[1]-v0
    if budget>=2**m:
        z=enumerate_masks(m);s=z.sum(1);ok=(s>0)&(s<m);z=z[ok].astype(float)
        v=np.asarray(game(z),dtype=float)-v0
        weights=np.array([(m-1)/(math.comb(m,int(k))*k*(m-k)) for k in s[ok]])
    else:
        rng=np.random.default_rng(seed);sizes=np.arange(1,m);prob=1/(sizes*(m-sizes));prob/=prob.sum();masks=[]
        while len(masks)<budget-2:
            row=np.zeros(m);size=int(rng.choice(sizes,p=prob));row[rng.choice(m,size,replace=False)]=1;masks.append(row)
            if pairing and len(masks)<budget-2:masks.append(1-row)
        z=np.asarray(masks);v=np.asarray(game(z),dtype=float)-v0;weights=np.ones(len(z))
    D=np.stack([np.prod(z[:,t],axis=1) for t in terms],axis=1)
    center=D-D.mean(axis=1,keepdims=True)
    residual=v-D.sum(axis=1)*(total/len(terms))
    theta=np.linalg.lstsq(center*np.sqrt(weights[:,None]),residual*np.sqrt(weights),rcond=1e-10)[0]
    theta=theta-theta.mean()+total/len(terms)
    phi=np.zeros(m)
    for coef,term in zip(theta,terms):phi[list(term)]+=coef/len(term)
    return phi
