"""Actual CPU PyTorch tests; these never imply FSDP/CUDA/Ray validation."""
import copy
import tempfile
from pathlib import Path
import unittest
import torch
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from toolrl_pipeline.io import read_jsonl, write_json
from toolrl_pipeline.tokenization import load_local_tokenizer
from toolrl_training.checkpoint import publish_checkpoint, validate_checkpoint, capture_rng, restore_rng
from toolrl_training.config import ROOT, load, grpo_overrides, sft_overrides
from toolrl_training.tensor_ops import causal_sft_loss


class TensorTests(unittest.TestCase):
    def test_real_veRL_sft_dataset_and_rl_dataset(self):
        from verl.utils.dataset.sft_dataset import SFTDataset
        from verl.utils.dataset.rl_dataset import RLHFDataset
        tok=load_local_tokenizer(ROOT/'data_local/tokenizers/Qwen2.5-1.5B-Instruct/989aa7980e4cf806f80c7fef2b1adb7bc71aa306')
        sft=SFTDataset(str(ROOT/'dataset/generated/phase2-v1/sft/train.parquet'),tok)
        val=SFTDataset(str(ROOT/'dataset/generated/phase2-v1/sft/validation.parquet'),tok)
        self.assertEqual((len(sft),len(val)),(245,37))
        for dataset in (sft,val):
            item=dataset[len(dataset)-1]
            self.assertEqual(item['labels'].shape,(3072,))
            active=(item['labels']!=-100)
            self.assertGreater(active.sum().item(),0)
            self.assertTrue(torch.all(item['labels'][item['attention_mask']==0]==-100))
        rl=RLHFDataset(str(ROOT/'dataset/generated/phase2-v1/grpo/train.parquet'),tok,max_prompt_length=4608,truncation='error')
        self.assertEqual(len(rl),2511)
        max_length=0
        for i in range(len(rl)):
            item=rl[i]
            self.assertEqual(item['split'],'train')
            self.assertIn(item['extra_info']['call_order'],('parallel','sequential'))
            self.assertEqual(item['input_ids'].shape,(4608,))
            max_length=max(max_length,item['attention_mask'].sum().item())
        self.assertEqual(max_length,4293)
        with self.assertRaises(ValueError):
            SFTDataset(str(ROOT/'dataset/generated/phase2-v1/sft/train.parquet'),tok,truncation='left')

    def test_gradient_accumulation_equals_full_token_weighted_batch(self):
        torch.manual_seed(5)
        logits=torch.randn(3,7,19,requires_grad=True)
        labels=torch.tensor([[-100,-100,3,4,5,6,-100],[-100,-100,-100,7,8,-100,-100],[-100,2,3,4,5,6,7]])
        total=causal_sft_loss(logits,labels)
        total.backward()
        expected=logits.grad.clone()
        split=logits.detach().clone().requires_grad_()
        denom=(labels[:,1:]!=-100).sum()
        for i in range(3):
            causal_sft_loss(split[i:i+1],labels[i:i+1],denom).backward()
        torch.testing.assert_close(split.grad,expected,rtol=0,atol=0)
        self.assertTrue(torch.all(expected[:2,:1]==0))
        self.assertGreater(expected[2,0].abs().sum().item(),0)

    def test_prompt_padding_labels_do_not_change_loss(self):
        torch.manual_seed(10)
        logits=torch.randn(1,6,13)
        labels=torch.tensor([[-100,-100,-100,3,4,-100]])
        before=causal_sft_loss(logits,labels)
        changed=logits.clone();changed[:,0:2]=100;changed[:,4:]=200
        self.assertEqual(before.item(),causal_sft_loss(changed,labels).item())
        with self.assertRaises(ValueError):
            causal_sft_loss(logits,torch.full_like(labels,-100))

    def test_actual_cpu_optimizer_scheduler_rng_resume(self):
        torch.manual_seed(7)
        model=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.Dropout(.2),torch.nn.Linear(4,1))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=.9)
        data=[torch.arange(8,dtype=torch.float32).reshape(2,4)+i for i in range(4)]
        def update(m,o,s,x):
            o.zero_grad();loss=m(x).square().mean();loss.backward();o.step();s.step()
            return loss.detach()
        update(model,optimizer,scheduler,data[0])
        with tempfile.TemporaryDirectory() as d:
            checkpoint=Path(d)/'runA/step1'
            def writer(p):
                torch.save(dict(weights=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                                rng=capture_rng(),progress={'step':1,'next_batch':1}),p/'state.pt')
            publish_checkpoint(checkpoint,writer,{'run_id':'runA','model_revision':'fixture'})
            expected_loss=update(model,optimizer,scheduler,data[1]);expected=copy.deepcopy(model.state_dict())
            restored=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.Dropout(.2),torch.nn.Linear(4,1))
            ro=torch.optim.AdamW(restored.parameters(),lr=.01);rs=torch.optim.lr_scheduler.StepLR(ro,1,.9)
            validate_checkpoint(checkpoint,{'run_id':'runA'})
            state=torch.load(checkpoint/'state.pt',weights_only=False)
            restored.load_state_dict(state['weights']);ro.load_state_dict(state['optimizer']);rs.load_state_dict(state['scheduler'])
            restore_rng(state['rng'])
            loss=update(restored,ro,rs,data[state['progress']['next_batch']])
            torch.testing.assert_close(loss,expected_loss,rtol=0,atol=0)
            for key,value in restored.state_dict().items():torch.testing.assert_close(value,expected[key],rtol=0,atol=0)
            self.assertEqual(rs.state_dict(),scheduler.state_dict())

    def test_actual_grpo_advantage_and_clipped_objective(self):
        from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage,compute_policy_loss
        rewards=torch.tensor([[0.,1.],[0.,3.],[0.,2.],[0.,2.]])
        mask=torch.ones_like(rewards)
        adv,_=compute_grpo_outcome_advantage(rewards,mask,['a','a','b','b'])
        self.assertTrue(torch.isfinite(adv).all())
        self.assertAlmostEqual(adv[0,0].item(),-2**-.5,places=5)
        self.assertTrue(torch.all(adv[2:]==0))
        old=torch.zeros((4,2));new=torch.tensor([[.5,.5]]*4,requires_grad=True)
        loss,clip,kl=compute_policy_loss(old,new,adv,mask,.2)
        loss.backward()
        self.assertTrue(torch.isfinite(new.grad).all())
        self.assertGreater(clip.item(),0)

    def test_hydra_parse_actual_vendored_configs(self):
        with initialize_config_dir(config_dir=str(ROOT/'verl/trainer/config'),version_base=None):
            c=load(ROOT/'configs/phase3/grpo_smoke.json')
            cfg=compose(config_name='ppo_trainer',overrides=grpo_overrides(c,'/local/model',ROOT/'dataset/generated/phase2-v1','/run/unique'))
            OmegaConf.resolve(cfg)
            self.assertEqual(cfg.actor_rollout_ref.rollout.n,4)
            self.assertEqual(cfg.algorithm.kl_ctrl.kl_coef,0)
            s=load(ROOT/'configs/phase3/sft_small.json')
            cfg=compose(config_name='sft_trainer',overrides=sft_overrides(s,'/local/model',ROOT/'dataset/generated/phase2-v1','/run/sft'))
            self.assertEqual(cfg.data.prompt_key,'prompt')
            self.assertTrue(cfg.model.enable_gradient_checkpointing)

class RewardManagerIntegrationTests(unittest.TestCase):
    def test_completion_only_reward_and_last_valid_token(self):
        import numpy as np
        from verl import DataProto
        from verl.trainer.main_ppo import RewardManager
        from toolrl_training.reward_probes import cases
        from toolrl_pipeline.protocol import render_completion
        from toolrl_pipeline.io import canonical
        tools,gt,_=cases()
        tok=load_local_tokenizer(ROOT/'data_local/tokenizers/Qwen2.5-1.5B-Instruct/989aa7980e4cf806f80c7fef2b1adb7bc71aa306')
        prompt=tok.encode('Tool definitions contain <tool_call> literals and unrelated instructions.',add_special_tokens=False)
        response=tok.encode(render_completion(gt),add_special_tokens=False)+[tok.eos_token_id]
        data=DataProto.from_dict(tensors={
            'prompts':torch.tensor([prompt]),
            'responses':torch.tensor([response+[tok.pad_token_id]*3]),
            'attention_mask':torch.tensor([[1]*(len(prompt)+len(response))+[0]*3])},
            non_tensors={
                'reward_model':np.array([{'ground_truth':render_completion(gt)}],dtype=object),
                'data_source':np.array(['rlla_phase2'],dtype=object),
                'available_tools':np.array([canonical(tools)],dtype=object),
                'extra_info':np.array([{'call_order':'parallel'}],dtype=object)})
        totals,formats,correctness,length=RewardManager(tok,0,'strict_matching')(data,1)
        self.assertEqual(totals.sum().item(),4)
        self.assertEqual(totals[0,len(response)-1].item(),4)
        self.assertEqual((formats.sum().item(),correctness.sum().item(),length.sum().item()),(1,3,0))
        self.assertEqual(totals[0,len(response):].sum().item(),0)
        data.batch['attention_mask'][:,len(prompt):]=0
        with self.assertRaises(ValueError):RewardManager(tok,0,'strict_matching')(data,1)

    def test_actual_reward_kl_zero_and_legacy_coefficient(self):
        from verl import DataProto
        from verl.trainer.ppo.ray_trainer import apply_kl_penalty
        from verl.trainer.ppo.core_algos import FixedKLController
        for coefficient in (0,.001):
            data=DataProto.from_dict(tensors={
                'responses':torch.ones((2,3),dtype=torch.long),
                'attention_mask':torch.ones((2,5)),
                'token_level_scores':torch.ones((2,3)),
                'old_log_probs':torch.ones((2,3)),
                'ref_log_prob':torch.zeros((2,3))})
            result,metrics=apply_kl_penalty(data,FixedKLController(coefficient))
            torch.testing.assert_close(result.batch['token_level_rewards'],torch.full((2,3),1-coefficient,dtype=torch.float32))
            self.assertEqual(metrics['critic/kl_coeff'],coefficient)
