# ToolRL: Reward is All Tool Learning Needs
[**🤗 Model**](https://huggingface.co/collections/emrecanacikgoz/toolrl-680706679204ead5a6d44f58) | [**📊 Dataset**](https://github.com/qiancheng0/ToolRL/tree/main/dataset) | [**📖 Paper**](https://arxiv.org/pdf/2504.13958)

![DataPipeline](assets/reward.png)

ToolRL is the code repository for paper "ToolRL: Reward is All Tool Learning Needs".

Our code is built upon [veRL](https://github.com/volcengine/verl) and [TinyZero](https://github.com/Jiayi-Pan/TinyZero).

## 本分支的低成本实验审计

中文 Phase 1 审计、架构图、数据/奖励/评测问题与 AutoDL 方案见 [docs/phase1](docs/phase1/README.md)。
本阶段未执行模型训练；原训练脚本及下方论文复现说明保留。原始依赖和数据划分需先审计验证，不建议直接启动长训练。
实验记录模板见 [experiment_record_template.yaml](docs/phase1/experiment_record_template.yaml)。
Bamboogle 凭据占位文件移至 `benchmarks/Bamboogle/secret.example.json`；运行前复制为本地被忽略的 `secret.json`，不要提交实际密钥。

## 🔍 Installation
Please install torch, vllm and ray according to your own environment configuration. We provide a configuration example adapted from TinyZero in the following:
```
# install torch
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip install vllm==0.6.3
pip install ray
```

Please further install the verl in the current project and flash attention.
```
# verl
pip install -e .

# flash attention 2
pip install flash-attn --no-build-isolation
```

## 📊 Dataset
We provide the raw dataset in `./dataset/rlla_4k_raw`, which consists of 2K ToolACE data, 1K Hammer (Masked) data, and 1K xLAM data.

The SFT data includes thought content distilled from Deepseek-R1, whereas the RL data contains only placeholders in the thought field.

For training purposes, the raw data must be further processed. The processed RL training data is available at `./dataset/rlla_4k`.


## 🧪 Training
For GRPO and PPO training, please specify the configuration in `./train_grpo.sh`, including the `BASE_MODEL` and `EXPERIMENT_NAME` variables. The dataset is set by default to `./dataset/rlla_4k`.
```
bash train_grpo.sh  # For GRPO Training
bash train_ppo.sh  # For PPO Training
```

### Reward variants
The training script uses by default the reward function introduced in Section 3.3 (\textit{Reward Design}) of the paper. In the following, we introduce several reward variants that can be activated via environment variables. All reward-specific environment variables are handled in the core reward module located at `./verl/utils/reward\_score/rlla.py`.
```
export WITHLENGTH=1 # Add the settled length reward function (Section 5.1)
export SCHEDULELENGTH=1 # Add the dynamic length reward function (Section 5.1)
export CORRECTMAX1=1 # Change to equal max (Section 5.2)
export MAX1STEP30MAX3=1 # Change to two stage scale (Section 5.2)
export SCHEDULEREWARD=1 # Chenge to smooth dynamic scale (Section 5.2)
export REFINEDREWARD=1 # Change to finegrained reward (Section 5.3)
export INTERMEDIATEREWARD=1 # Change to intermediate reward (Section 5.3)
export COARSEREWARD=1 # Change to coarse reward (Section 5.3)
```

To train the model with a specific reward variant, please enable one of the environment variables described above.


## 🖊️ Citation
```text
@article{qian2025toolrl,
  title={ToolRL: Reward is All Tool Learning Needs},
  author={Qian, Cheng and Acikgoz, Emre Can and He, Qi and Wang, Hongru and Chen, Xiusi and Hakkani-T{\"u}r, Dilek and Tur, Gokhan and Ji, Heng},
  journal={arXiv preprint arXiv:2504.13958},
  year={2025}
}
```
