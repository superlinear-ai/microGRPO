# microGRPO

A tiny single-file implementation of Group Relative Policy Optimization (GRPO) as introduced by the [DeepSeekMath paper](https://arxiv.org/abs/2402.03300).

For further reading on GRPO, see [Yuge (Jimmy) Shi's blog post](https://yugeten.github.io/posts/2025/01/ppogrpo/) and [Nathan Lambert's RLHF book](https://rlhfbook.com/c/11-policy-gradients.html).

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

You should see that the policy learns to improve its average score from around 17% to about 48% over 2000 iterations.

## Background

#### File structure

The file is structured into five sections:

1. 🕹️ Game (~50 lines): An implementation of the Battleship board game
2. 🌍 Environment (~60 lines): The API with which an agent can interact with the game
3. 🧠 Policy (~40 lines): A model that produces action probabilities given the observed environment state
4. 🎯 GRPO (~90 lines): The GRPO objective function and training data generator
5. ⚡ Train (~40 lines): The loop that collects training data and optimizes the GRPO objective with AdamW

#### GRPO config

Starting a training run requires defining a `GRPOConfig` with your choice of environment (here, `BattleshipEnv`), a function that evaluates the policy model given its parameters (here, `neural_battleship_policy`), and another function that evaluates a reference policy model that you don't want the policy to deviate too much from (here, `reference_battleship_policy`):

```python
# Define the environment, the policy model to optimize, and a reference policy model.
grpo_config = GRPOConfig(
    environment=BattleshipEnv,
    policy=neural_battleship_policy,
    reference_policy=reference_battleship_policy,
)

# Initialize the policy model parameters.
θ_init = neural_battleship_policy_init()

# Train the policy model by maximizing the GRPO objective with AdamW.
θ_star, rewards_val = train_grpo(AdamWOptimizer(θ_init, learning_rate=3e-4), grpo_config)
```
