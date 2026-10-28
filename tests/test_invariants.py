from pathlib import Path
import itertools,sys,unittest
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from common import Frozen,enumerate_masks,exact_shapley
from cv_shapley import cv_shapley

class AttributionInvariants(unittest.TestCase):
    def test_exact_subset_sum_matches_all_permutations(self):
        m=4;z=enumerate_masks(m)
        values=z@np.array([.1,-.2,.5,0])+2*z[:,0]*z[:,1]+3*z[:,0]*z[:,2]*z[:,3]
        expected=np.zeros(m)
        for order in itertools.permutations(range(m)):
            coalition=0
            for player in order:
                expected[player]+=values[coalition|(1<<player)]-values[coalition]
                coalition|=1<<player
        expected/=24
        np.testing.assert_allclose(exact_shapley(values,m),expected,atol=1e-13,rtol=0)

    def test_zero_endpoint_game_keeps_a_true_null_player(self):
        m=6;z=enumerate_masks(m);values=(z[:,:5].sum(1)==2).astype(float)
        def game(masks):return values[(np.asarray(masks,dtype=int)*(1<<np.arange(m))).sum(1)]
        for seed in [3,7,11]:
            phi,_,info=cv_shapley(game,m,16,seed)
            self.assertLess(abs(phi[-1]),1e-12)
            self.assertLess(abs(phi.sum()),1e-12)
            self.assertLessEqual(info['theoretical_query_budget'],16*(m+1))

    def test_additive_target_and_query_accounting(self):
        weights=np.array([.3,-.1,.7,0,.2])
        class Game:
            def __init__(self):self.rows=0
            def __call__(self,masks):self.rows+=len(masks);return np.asarray(masks)@weights
        game=Game();phi,_,info=cv_shapley(game,5,16,4)
        np.testing.assert_allclose(phi,weights,atol=1e-12,rtol=0)
        self.assertEqual(game.rows,info['theoretical_query_budget'])
        self.assertLessEqual(game.rows,16*6)

    def test_documented_argument_domain(self):
        for m,budget in [(2,16),(4,8),(4,18)]:
            with self.assertRaises(ValueError):cv_shapley(lambda z:np.zeros(len(z)),m,budget,1)

    def test_grouping_preserves_dense_probabilities(self):
        torch.manual_seed(37);net=Frozen().double().eval()
        dense=torch.zeros(1,13,463,dtype=torch.float64)
        dense[0,:3]=torch.randn(3,463,dtype=torch.float64)
        grouped=torch.cat([dense[:,:3],dense[:,3:4]],dim=1)
        counts=torch.tensor([1,1,1,10],dtype=torch.float64)
        with torch.no_grad():a=net.original_forward(dense)[0];b=net(grouped,counts)[0]
        reconstructed=torch.cat([b[:3],b[3:].repeat(10)/10])
        torch.testing.assert_close(a,reconstructed,atol=1e-12,rtol=1e-12)

if __name__=='__main__':unittest.main()
