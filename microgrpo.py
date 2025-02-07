#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.13"
# dependencies = ["autograd~=1.7.0", "numpy~=2.2.2", "tqdm~=4.67.1"]
# ///

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Callable, TypeAlias

import autograd.numpy as np  # This is only a thin wrapper around NumPy...
from autograd import grad  # ...to enable automatic differentation with `grad`.
from tqdm.auto import tqdm

# --- Game ---


# 🌊 = no ship present
# 🚢 = ship present
# 💦 = missile miss
# 💥 = missile hit
BattleshipGameBoard: TypeAlias = np.ndarray[tuple[int, int], np.str_]


@dataclass
class BattleshipGameRules:
    board_size: int = 5  # A 5 x 5 game board
    ships: tuple[int, ...] = (0, 0, 1, 1, 1, 0)  # 1x destroyer, 1x cruiser/submarine, 1x battleship


@dataclass
class BattleshipGame:
    board: BattleshipGameBoard
    rules: BattleshipGameRules = field(default_factory=BattleshipGameRules)

    @staticmethod
    def random_board(rules: BattleshipGameRules | None = None, seed: int | None = None) -> BattleshipGameBoard:
        rules = rules or BattleshipGameRules()
        random_state = np.random.RandomState(seed)
        ships = [ship_size for ship_size, ship_count in enumerate(rules.ships) for _ in range(ship_count)]
        ships_placed = False
        while not ships_placed:
            board = np.full((rules.board_size, rules.board_size), "🌊", dtype=np.str_)
            for ship_size in ships:
                if ship_size > rules.board_size:
                    return
                ship_top_left = random_state.randint(low=0, high=rules.board_size - (ship_size - 1), size=2)
                ship_bottom_right = ship_top_left + 1
                ship_bottom_right[random_state.randint(low=0, high=2)] += ship_size - 1
                if np.all(board[ship_top_left[0] : ship_bottom_right[0], ship_top_left[1] : ship_bottom_right[1]] == "🌊"):
                    board[ship_top_left[0] : ship_bottom_right[0], ship_top_left[1] : ship_bottom_right[1]] = "🚢"
                else:
                    break
            else:
                ships_placed = True
        return board

    def play(self, fire: tuple[int, int]) -> bool:
        hit = self.board[fire] in ("🚢", "💥")
        self.board[fire] = "💥" if hit else "💦"
        return hit

    def score(self) -> float:
        done = not np.any(self.board == "🚢")
        efficiency: float = 1.0 - np.sum(self.board == "💦") / (self.board.size - np.sum(self.board == "💥") + 1) if done else 0.0
        return efficiency

    def __repr__(self) -> str:
        return "\n".join("".join(row) for row in self.board)


# --- Environment ---

ActionProbaArray: TypeAlias = np.ndarray[tuple[int], np.float32]
ObservationArray: TypeAlias = np.ndarray[tuple[int, ...], np.float32]


class Environment(ABC):
    max_steps: int

    @property
    @abstractmethod
    def observation(self) -> ObservationArray:
        pass

    @classmethod
    @abstractmethod
    def reset(cls, init_seed: int | None = None, step_seed: int | None = None) -> tuple["Environment", ObservationArray]:
        pass

    @abstractmethod
    def sample_action(self, action_proba: ActionProbaArray) -> int:
        pass

    @abstractmethod
    def step(self, action: int) -> tuple[ObservationArray, float, bool]:
        pass


class BattleshipEnv(Environment):
    rules = BattleshipGameRules()
    max_steps = rules.board_size**2

    def __init__(self, init_seed: int | None = None, step_seed: int | None = None) -> None:
        self.state = BattleshipGame(board=BattleshipGame.random_board(self.rules, init_seed), rules=self.rules)
        self.random_state = np.random.RandomState(step_seed)

    @property
    def observation(self) -> ObservationArray:
        # 0 = fog of war, -1 = missile miss, 1 = missile hit
        encoded_board = np.zeros(self.state.board.shape, dtype=np.float32)
        encoded_board[self.state.board == "💦"] = -1.0
        encoded_board[self.state.board == "💥"] = 1.0
        return encoded_board

    @classmethod
    def reset(cls, init_seed: int | None = None, step_seed: int | None = None) -> tuple["BattleshipEnv", ObservationArray]:
        env = cls(init_seed, step_seed)
        return env, env.observation

    def sample_action(self, action_proba: ActionProbaArray) -> int:
        # Mask out illegal actions.
        illegal_actions = np.ravel(self.observation != 0.0)
        action_proba[illegal_actions] = 0.0
        action_proba /= np.sum(action_proba)
        # Sample an action from the probability distribution.
        action = int(self.random_state.choice(len(action_proba), p=action_proba))
        return action

    def step(self, action: int) -> tuple[ObservationArray, float, bool]:
        self.state.play(fire=divmod(action, self.state.rules.board_size))
        reward = self.state.score()
        done = reward > 0.0
        return self.observation, reward, done


# --- Policy ---


ParamsDict: TypeAlias = dict[str, np.ndarray[tuple[int, ...], np.float32]]


def neural_battleship_policy_init(rules: BattleshipGameRules | None = None, seed: int = 42) -> ParamsDict:
    rules = rules or BattleshipGameRules()
    num_tiles = rules.board_size**2
    random_state = np.random.RandomState(seed)
    scale = np.sqrt(2.0 / (2 * num_tiles))  # Xavier/Glorot initialization
    params = {
        "W1": random_state.normal(scale=scale, size=(num_tiles, num_tiles)).astype(np.float32),
        "b1": np.zeros(num_tiles, dtype=np.float32),
        "W2": random_state.normal(scale=scale, size=(num_tiles, num_tiles)).astype(np.float32),
        "b2": np.zeros(num_tiles, dtype=np.float32),
    }
    return params


def neural_battleship_policy(params: ParamsDict, observation: ObservationArray) -> ActionProbaArray:
    # A simple feedforward neural network with a single hidden layer.
    x = np.ravel(observation)
    h = np.tanh(params["W1"] @ x + params["b1"])
    logits = params["W2"] @ h + params["b2"]
    logits -= np.max(logits)  # Softmax is invariant to shifting the logits.
    exp_logits = np.exp(logits)
    softmax = exp_logits / np.sum(exp_logits)
    return softmax


def reference_battleship_policy(observation: ObservationArray) -> ActionProbaArray:
    # Fire on any fog of war tile with uniform probability.
    p = np.ravel((observation == 0.0).astype(np.float32)) + np.sqrt(np.finfo(np.float32).eps)
    p = p / np.sum(p)
    return p


# --- GRPO ---


ActionArray: TypeAlias = np.ndarray[tuple[int], np.intp]
RewardArray: TypeAlias = np.ndarray[tuple[int], np.float32]
AdvantageArray: TypeAlias = np.ndarray[tuple[int], np.float32]
PolicyFunction: TypeAlias = Callable[[ParamsDict, ObservationArray], ActionProbaArray]
ReferencePolicyFunction: TypeAlias = Callable[[ObservationArray], ActionProbaArray]
Group: TypeAlias = tuple[list[list[ObservationArray]], list[ActionProbaArray], list[ActionArray], RewardArray, AdvantageArray]


@dataclass
class GRPOConfig:
    environment: type[Environment]
    policy: PolicyFunction
    reference_policy: ReferencePolicyFunction

    ε: float = 0.9  # Policy ratio clip epsilon
    ß: float = 0.0  # Weight for KL divergence between the policy and the reference policy
    G: int = 16  # Number of trajectories per group
    B: int = 4  # Number of groups per mini-batch
    M: int = 2048  # Number of mini-batches to train on
    μ: int = 10  # Number of gradient steps per mini-batch


def collect_group(policy_params: ParamsDict, grpo_config: GRPOConfig, env_seed: int | None = None) -> Group:
    # Initialize the group output.
    group_observations: list[list[ObservationArray]] = [[] for _ in range(grpo_config.G)]
    group_actions = [np.empty(grpo_config.environment.max_steps, dtype=np.intp) for _ in range(grpo_config.G)]
    group_actions_proba = [np.empty(grpo_config.environment.max_steps, dtype=np.float32) for _ in range(grpo_config.G)]
    group_rewards = np.zeros(grpo_config.G, dtype=np.float32)
    # Create a fixed environment initialization seed.
    init_seed = env_seed if env_seed is not None else np.random.randint(2**32)
    # Generate trajectories starting from the initial environment.
    for group in range(grpo_config.G):
        # Start a new environment (a game) from a fixed initial seed.
        env, observation = grpo_config.environment.reset(init_seed=init_seed, step_seed=init_seed * group)
        for step in range(env.max_steps):
            # Evaluate the policy model to obtain the action probability distribution.
            action_proba = grpo_config.policy(policy_params, observation)
            # Sample an action from the policy's action probability distribution.
            action = env.sample_action(action_proba)
            # Update the group output.
            group_observations[group].append(observation)
            group_actions[group][step] = action
            group_actions_proba[group][step] = action_proba[action]
            # Advance the environment with the sampled action.
            observation, reward, done = env.step(action)
            # Check if this trajectory is done.
            if done:
                group_rewards[group] = reward  # GRPO only considers the terminal reward.
                break
    # Compute the GRPO advantages across the group, but assign them to the actions within each trajectory.
    group_advantages = (group_rewards - np.mean(group_rewards)) / max(np.std(group_rewards), np.finfo(np.float32).eps)
    return (group_observations, group_actions_proba, group_actions, group_rewards, group_advantages)


def grpo_objective(policy_params: ParamsDict, group: Group, grpo_config: GRPOConfig) -> float:
    # For each trajectory in the given group...
    grpo = 0.0
    for observations, actions_proba, actions, _, advantage in zip(*group):
        # ...accumulate the trajectory's step contributions to the GRPO objective.
        for observation, π_θ_t_old, action in zip(observations, actions_proba, actions):
            π_θ_t = grpo_config.policy(policy_params, observation)[action]
            π_ref_t = grpo_config.reference_policy(observation)[action]
            ratio = π_θ_t / π_θ_t_old
            clipped_ratio = np.clip(π_θ_t / π_θ_t_old, 1 - grpo_config.ε, 1 + grpo_config.ε)
            grpo += min(ratio * advantage, clipped_ratio * advantage) / len(actions)  # Advantage
            grpo += -grpo_config.ß * (π_ref_t / π_θ_t - np.log(π_ref_t / π_θ_t) - 1) / len(actions)  # KL divergence
    grpo /= grpo_config.G
    grpo = -grpo  # Flip the sign to turn the maximization problem into a minimization problem.
    return grpo


# --- Train ---


class AdamWOptimizer:
    def __init__(self, params: ParamsDict, learning_rate: float = 3e-4, ß1: float = 0.9, ß2: float = 0.999, ε: float = 1e-8, λ: float = 0.01) -> None:
        self.params = params
        self.learning_rate = learning_rate
        self.ß1 = ß1
        self.ß2 = ß2
        self.ε = ε
        self.λ = λ
        self.t = 1
        self.state = {key: {"m": np.zeros_like(value), "v": np.zeros_like(value)} for key, value in params.items()}

    def step(self, grad: ParamsDict) -> None:
        for key in self.params:
            self.state[key]["m"] = self.ß1 * self.state[key]["m"] + (1 - self.ß1) * grad[key]
            self.state[key]["v"] = self.ß2 * self.state[key]["v"] + (1 - self.ß2) * (grad[key] ** 2)
            m_hat = self.state[key]["m"] / (1 - self.ß1**self.t)
            v_hat = self.state[key]["v"] / (1 - self.ß2**self.t)
            update = self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.ε) + self.λ * self.params[key])
            self.params[key] -= update
        self.t += 1


def train_grpo(optimizer: AdamWOptimizer, grpo_config: GRPOConfig) -> tuple[ParamsDict, RewardArray]:
    # Define the GRPO objective for a mini-batch of groups of trajectories.
    grpo_objective_batch = lambda policy_params, groups, grpo_config: sum(grpo_objective(policy_params, group, grpo_config) for group in groups)  # noqa: E731
    # Define the gradient of the GRPO objective w.r.t. the policy parameters (the first argument of grpo_objective).
    grpo_objective_batch_grad = grad(grpo_objective_batch)
    rewards_val = np.zeros(grpo_config.M, dtype=np.float32)
    for iter in (pbar := tqdm(range(grpo_config.M))):
        # Collect a mini-batch of groups of trajectories to learn from.
        groups = [collect_group(optimizer.params, grpo_config, env_seed=(iter + 1) * 128 + i) for i in range(grpo_config.B)]
        # Optimize the GRPO objective determined by the current mini-batch for a few steps.
        for _ in range(grpo_config.μ):
            # Compute the gradient and update the solution.
            optimizer.step(grpo_objective_batch_grad(optimizer.params, groups, grpo_config))
        # Track progress of the validation reward.
        groups_val = [collect_group(optimizer.params, replace(grpo_config, G=8), env_seed=i) for i in range(64)]
        rewards_val[iter] = sum(np.mean(group_val[3]) for group_val in groups_val) / len(groups_val)
        pbar.set_description(f"reward_val={rewards_val[iter]:.3f}")
    return optimizer.params, rewards_val


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
# Save the trained policy model parameters and the validation rewards.
np.savez("battleship_policy.npz", **θ_star)
np.savetxt("battleship_rewards.csv", rewards_val, delimiter=",")
