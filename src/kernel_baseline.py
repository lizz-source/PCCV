from common import *
import math
def kernel_shap(game,m,budget,seed):
    if budget>=2**m:
        masks=enumerate_masks(m);values=game(masks);use=masks.sum(1);ok=(use>0)&(use<m);z=masks[ok].astype(float);v=values[ok];total=float(values[-1])
        weights=np.array([(m-1)/(math.comb(m,int(s))*s*(m-s)) for s in use[ok]])
        Q=(z[:,:-1]-z[:,-1:])*np.sqrt(weights[:,None]);y=(v-z[:,-1]*total)*np.sqrt(weights)
    else:
        rng=np.random.default_rng(seed);sizes=np.arange(1,m);prob=1/(sizes*(m-sizes));prob/=prob.sum();zs=[]
        for _ in range((budget-1)//2):
            size=int(rng.choice(sizes,p=prob));row=np.zeros(m);row[rng.choice(m,size,replace=False)]=1;zs.extend([row,1-row])
        z=np.asarray(zs);v=game(z);total=float(game(np.ones((1,m)))[0]);Q=z[:,:-1]-z[:,-1:];y=v-z[:,-1]*total
    a=np.linalg.lstsq(Q,y,rcond=None)[0];return np.r_[a,total-a.sum()]
