# Loss-Advantage Mismatch

> 本文件是飞书云文档 [Loss-Advantage Mismatch](https://my.feishu.cn/docx/CWmadoODHongjQxCS9nc32rinnb) 的本地副本。


## Implementation 细节

- Minibatch size 决定是否on-policy (可以视为： 一次minibatch 一次梯度更新， 每次minibatch要用policy model 重新算一次概率）, micro-batch 只是为了减轻显卡压力，分多次放进去求平均（通过gradient_accumulation来实现) ，但traj Loss 要求Micro-batch 里面有完整的traj。 存在多次切分导致bug：
  - 1.FSDP 切分rollout 的 数据 (又一堆steps堆叠) 到n个GPU上 （平均切分， 导致有些traj断掉）
  - 2.切Minibatch
  - 3.切microbatch
- 需要保证这三层切分都不切断traj。 但是1根本很难做到， 2 和 3 要做到的话也需要解决"分布式死锁问题"：即不同GPU的代码路径需要一模一样（比如for循环次数）。
  - 2导致的分布式死锁问题可以通过完全on policy 解决（只有一个minibatch)
  - 3导致的分布式死锁问题可以通过padding解决（找到最大micro batch size的worker， 其他worker 加padding）
  - 1似乎没法解决。
- （2026.3.1） 新问题： GTPO 的 ppo_kl 很低 模型entropy loss 很低 ，actor/ppo_kl 基本为0（而其他正常的实验都大于0），actor/pg_clipfrac 也基本为0（其他正常的实验都大于0）。 ---> **一个Minibatch <=> 一次更新**，traj loss setting下 一轮只有一个minibatch（因为纯 on policy）， 而其他setting 有多个 -> 更新一次和n次的区别，所以更新得慢。
  - 方案A: 减小loss的分母 （用处不大）
    - PPO clip 是信任域机制：多步训练中，后面的 epoch ratio 逐渐偏离 1 → clip 开始生效 → 自动限制更新幅度。方案 A 的一大步没有这个保护（先取值，再算梯度，取值之后clip就没了），可能直接跳到 loss landscape 的不好区域。
    - Loss landscape 非线性：RL 的 loss 高度非线性（advantages 的含义随策略变化而变化）。多步能"顺着弯走"，考虑loss landscape的曲率，一大步可能"冲过头"。
    - Adam 自适应：Adam 需要多步来调整 momentum 和 learning rate。1 步给不了 Adam 自适应的机会。
  - 方案B: 增大PPO epoch 数 （效果上更等效，但慢n倍），但可以保证不同实验的更新次数是一样
    - （2026.3.1）已采用该方案
    - (2026.3.9) 能训动了（kl_loss 正常了), 但是很波动，performance差。原因分析是： 同一批大样本连续更新16次vs 大样本拆成16份， 一份更新一次。 前者不正常， 后者正常。
      - 现象分析： valid_action_ratio 低（0.85 左右), 且entropy loss高
      - 原因： **entropy loss 和kl loss 没有被正确缩放，偏大---> 修复了之后work了**
    - 2026.3.11 修复之后entropy loss 急剧降低（熵崩塌）， kl loss 很大 极不稳定
      - 现象：一次实验val success rate上升很快， 另一次基本不上升 （熵崩塌导致）
      - 问题分析： **同一批大样本连续更新16次vs 大样本拆成16份， 一份更新一次。 前者不正常， 后者正常。 （我的理解： 前者的方式可以很好地拟合到这一批数据(导致了过拟合）（原因： 梯度方差太小， 过于稳定）， 但后者的方式对其他数据泛化性更好，因为后者的梯度方差更大， 天然正则化）**

## 实验计划

adv 和 loss 对齐，在多场景下 (纯math 4个实验：2 loss × 3 adv (GRPO token + GRPO step + PPO_adv) = 6 个实验；agent场景两个benchmark 每个bench 12个实验： 3×3 + gigpo 固定adv为step，3个 loss）

- Math:
  - 2.27 **还差PPO的两个Loss**， 并且其他的多跑两个seed。
- Agent:
  - Alfworld
    - 2.27 已完成8个， 还差traj Loss 的 四个实验
      - 3.15 【 traj loss + 3 个 adv 】重新跑了 + [GiGPO_adv + step_loss] 重新跑了（已经work） [GiGPO_adv + token_loss] 补了两个seed （完成。）
  - Webshop
    - 2.27 已完成8个， 还差traj Loss 的 四个实验
      - 现有问题， **还是GIGPO 对齐和不对齐效果差不多**。

## 实验进度 Track 表格 (Claude Code 自动维护, 更新于 2026-03-21 18:00)

- 核心实验指标： val/success_rate
- 每一个实验需要3个seed
- 期望实验结果：Loss-Advantage 对齐时效果最好（即对角线最优）
  - A_step × L_step > A_step × L_token, A_step × L_traj
  - A_token × L_token > A_token × L_step, A_token × L_traj
  - A_traj × L_traj > A_traj × L_token, A_traj × L_step
  - A_gigpo × L_step > A_gigpo × L_token, A_gigpo × L_traj（gigpo adv 粒度为 step）
- 表格中 ⭐ 标记期望最优的对角线格子
- 数据来源：wandb 有效可见 runs（排除 tag 过滤 + eye-icon 隐藏的 runs）

### ALFWorld 1.5B

- 主要看第150轮（末尾）时候的表现

| Adv \ Loss | L_token (vanilla) | L_step (gspo) | L_traj (gtpo) | 状态 |
|---|---|---|---|---|
| **A_step** | 2 fin + 2 crash, best=0.77 | ⭐ 🔄 已提交 Phase 1 (3 seeds) | 🔄 已提交 Phase 2 (3 seeds) | L_token 补 1 seed (Phase 2) |
| **A_token** | ⭐ 3 fin ✅, best=0.75 | 🔄 已提交 Phase 1 (3 seeds) | 🔄 已提交 Phase 2 (3 seeds) | Phase 1+2 已提交 |
| **A_traj** | 3 fin ✅, best=0.52 | 🔄 已提交 Phase 1 (3 seeds) | ⭐ 1 fin + 2 crash, best=0.44 | Phase 1+2 已提交 |
| **A_gigpo** | 3 fin ✅, best=0.89 | ⭐ 1 fin + 1 crash + 2 run, best=0.91 | 3 fin + 2 crash, best=0.33 | L_step 正在跑 |
| *说明* | *clip=0.2 ✅* | *Phase 1: clip=0.2 重跑* | *Plan B (4 epochs)* | *⭐=期望对角线最优* |

### WebShop 1.5B

| Adv \ Loss | L_token (vanilla) | L_step (gspo) | L_traj (gtpo) | 状态 |
|---|---|---|---|---|
| **A_step** | 4 fin + 1 crash ✅, best=0.74 | ⭐ 🔄 Phase 3 已提交 (3 seeds) | 🔄 Phase 4 已提交 (3 seeds) | Phase 3+4 已提交 |
| **A_token** | ⭐ 3 fin ✅, best=0.76 | 🔄 Phase 3 已提交 (3 seeds) | 🔄 Phase 4 已提交 (3 seeds) | Phase 3+4 已提交 |
| **A_traj** | 3 fin + 2 crash ✅, best=0.75 | 🔄 Phase 3 已提交 (3 seeds) | ⭐ 🔄 Phase 4 已提交 (3 seeds) | Phase 3+4 已提交 |
| **A_gigpo** | 4 fin ✅, best=0.74 | ⭐ 🔄 Phase 3 已提交 (3 seeds) | 🔄 Phase 4 已提交 (3 seeds) | Phase 3+4 已提交 |
| *说明* | *clip=0.2 ✅* | *旧 CLIP BUG runs 已标 meaningless* | *Plan B (4 epochs)* | *⭐=期望对角线最优* |
