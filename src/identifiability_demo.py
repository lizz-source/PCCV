from common import *
rng=np.random.default_rng(3101);N=8;D=4;F=rng.lognormal(0,.4,(N,D));w=rng.normal(0,.2,(N,D));P=enumerate_masks(N)[1:];X=P[:,:,None]*F
def predict(x,lam):
    present=x[:,:,1]>0
    logits=(x*w).sum(-1)+lam*x[:,:,1]*(F[None,:,0]-x[:,:,0])
    logits=np.where(present,logits,-1e30);p=np.exp(logits-logits.max(1,keepdims=True))*present;return p/p.sum(1,keepdims=True)
base=predict(X,0);perturbed=X.copy();perturbed[:,:,0]=0;reference=predict(perturbed,0)
rows=[]
for lam in [-2,-1,0,1,2]:
    obs=predict(X,lam);after=predict(perturbed,lam)
    rows.append({'lambda':lam,'observational_max_probability_difference':float(np.max(np.abs(obs-base))),'intervention_mean_bc_vs_reference_predictor':float(bc(after,reference).mean()),'mean_own_pathway_deletion_bc':float(bc(after,base).mean())})
table('results/observational_equivalence_demo.csv',rows)
np.savez_compressed(ROOT/'results/observational_equivalence_demo.npz',F=F,w=w,all_observed_memberships=P)
assert max(r['observational_max_probability_difference'] for r in rows)<1e-12
assert max(r['intervention_mean_bc_vs_reference_predictor'] for r in rows)>.05
dump('results/identifiability_verification.json',{'observed_membership_patterns':len(P),'predictors':5,'observational_equality_max_error':max(r['observational_max_probability_difference'] for r in rows),'largest_intervention_mean_bc_disagreement':max(r['intervention_mean_bc_vs_reference_predictor'] for r in rows),'pass':True})
print(rows,flush=True)
