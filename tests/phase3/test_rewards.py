import itertools
import unittest
from toolrl_pipeline.protocol import render_completion
from toolrl_pipeline.verifier import verify
from toolrl_training.rewards import compute_reward
from toolrl_training.reward_probes import cases, run_probes


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.tools,self.gt,_=cases()
    def score(self,g,p,order='parallel'):
        return compute_reward('strict_matching',render_completion(p),render_completion(g),self.tools,{'call_order':order})
    def test_adversarial_independent_verifier_agreement(self):
        report=run_probes()
        self.assertEqual(report['strict_fullscore_false_positives'],0)
        self.assertEqual(report['legacy_fullscore_false_positives'],1)
        self.assertEqual(self.score(self.gt,self.gt),(4,1,3,0))
        with self.assertRaises(ValueError):
            compute_reward('unknown','','',[])
    def test_partial_credit_and_extra_arguments(self):
        wrong=[{'name':'f','parameters':{'x':2}}]
        extra=[{'name':'f','parameters':{'x':1,'junk':9}}]
        self.assertGreater(self.score(self.gt,wrong)[0],-3)
        self.assertLess(self.score(self.gt,wrong)[0],4)
        self.assertLess(self.score(self.gt,extra)[0],4)
        self.assertEqual(self.score(self.gt,extra)[1],0)
    def test_duplicate_global_assignment_and_order(self):
        self.tools[0]['parameters']['properties']['y']={'type':'integer'}
        g=[{'name':'f','parameters':{'x':1,'y':1}},{'name':'f','parameters':{'x':1,'y':2}}]
        p=[{'name':'f','parameters':{'x':1,'y':2}},{'name':'f','parameters':{'x':2,'y':1}}]
        scores={self.score(list(a),list(b))[0] for a in itertools.permutations(g) for b in itertools.permutations(p)}
        self.assertEqual(len(scores),1)
        self.assertEqual(self.score(g,list(reversed(g)))[0],4)
        self.assertLess(self.score(g,list(reversed(g)),'sequential')[0],4)
        self.assertLess(self.score(g,g+[g[0]])[0],4)
    def test_parameter_dict_order_invariance(self):
        self.tools[0]['parameters']['properties']['y']={'type':'integer'}
        g=[{'name':'f','parameters':{'x':1,'y':2}}]
        p=[{'parameters':{'y':2,'x':1},'name':'f'}]
        self.assertEqual(self.score(g,p)[0],4)
    def test_free_text_no_invented_semantic_reward(self):
        good=compute_reward('strict_matching',render_completion([],response='wrong answer'),render_completion([]),[])
        self.assertEqual(good,(1,1,0,0))
        self.assertLess(self.score([],self.gt)[0],good[0])
    def test_fullscore_requires_em_small_exhaustive(self):
        g=self.gt
        predictions=[[],g]+[[{'name':name,'parameters':{'x':x}}] for name in ('f','z') for x in (True,1,'1',2,None)]
        predictions += [g+[p[0]] for p in predictions if p]
        for p in predictions:
            score=self.score(g,p)[0]
            if score>=4-1e-9:
                self.assertTrue(verify(g,render_completion(p),self.tools,{'call_order':'parallel'})['metrics']['exact_match_accuracy'])

class RealTrainingRewardSanityTests(unittest.TestCase):
    def test_all_real_rl_train_val_targets_schema_and_partial_reward(self):
        from toolrl_pipeline.io import read_jsonl
        from toolrl_training.config import ROOT
        count=0
        for split in ('train','validation'):
            for row in read_jsonl(ROOT/f'dataset/generated/phase2-v1/rl/{split}.jsonl'):
                scores=compute_reward('strict_matching',row['target_output'],row['target_output'],row['tools'],row['metadata'])
                self.assertAlmostEqual(scores[0],4 if row['ground_truth'] else 1)
                self.assertEqual(scores[3],0)
                count+=1
        self.assertEqual(count,2818)
