# microGRPO

A tiny single-file implementation of Group Relative Policy Optimization (GRPO) as introduced by the DeepSeekMath paper[^1][^2][^3].

🆕 microGRPO now implements the GRPO improvements introduced by Dr. GRPO[^4], Apple's LOOP[^5], and Mistral's Magistral[^6]:
1. 💥 Remove per-group advantage normalization[^4]
2. ⛳️ Leave-one-out advantage[^5] (LOOP only)
3. 🔥 Eliminate KL divergence[^5]
4. 🎢 Normalize loss[^5]
5. 🏆 Add per-batch advantage normalization[^6] (Magistral only)
6. 🚦 Relax trust region bounds[^5]
7. 🌈 Eliminate non-diverse groups[^5]

[^1]: [The DeepSeekMath paper](https://arxiv.org/abs/2402.03300)
[^2]: [Yuge (Jimmy) Shi's blog post](https://yugeten.github.io/posts/2025/01/ppogrpo/)
[^3]: [Nathan Lambert's RLHF book](https://rlhfbook.com/c/11-policy-gradients.html)
[^4]: [The Dr. GRPO paper](https://arxiv.org/pdf/2503.20783)
[^5]: [Apple's LOOP paper](https://arxiv.org/pdf/2502.01600)
[^6]: [Mistral's Magistral paper](https://arxiv.org/pdf/2506.10910)

## Features

1. 🐭 Only ~300 lines of code
2. 📦 In pure NumPy, with [autograd](https://github.com/HIPS/autograd) to compute the gradient
3. ✅ Type annotated and linted
4. ✂️ Easily swap out the default game and train on any other game or environment

## Getting started

> [!NOTE]
> You'll need to [install uv](https://docs.astral.sh/uv/getting-started/installation/) to run the commands below.

To start teaching a policy to play a simplified version of [Battleship](https://en.wikipedia.org/wiki/Battleship_(game)), run:
```sh
uv run microgrpo.py
```

You should see that the policy learns to improve its average score from around 15% to about 50% over 2000 iterations:

![Battleship policy trained with GRPO](https://github.com/user-attachments/assets/de464264-2d1c-43f2-9bc3-dcd9eea48c45)

## Background

#### File structure

The file is structured into five sections:

1. 🕹️ Game (~50 lines): An implementation of the Battleship board game
2. 🌍 Environment (~60 lines): The API with which an agent can interact with the game
3. 🧠 Policy (~30 lines): A model that produces action probabilities given the observed environment state
4. 🎯 GRPO (~80 lines): The GRPO objective function and training data generator
5. ⚡ Train (~50 lines): The loop that collects training data and optimizes the GRPO objective with AdamW

#### GRPO config

Starting a training run only requires defining a `GRPOConfig` with your choice of environment (here, `BattleshipEnv`) and a function that evaluates the policy model given its parameters (here, `neural_battleship_policy`):

```python
# Define the environment and the policy model to optimize.
grpo_config = GRPOConfig(environment=BattleshipEnv, policy=neural_battleship_policy)

# Train the policy model by maximizing the GRPO objective with AdamW.
θ_star, rewards_val = train_grpo(θ_init := neural_battleship_policy_init(), grpo_config)
```
