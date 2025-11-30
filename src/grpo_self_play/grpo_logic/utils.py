import torch
import torch.nn.functional as F
from dataclasses import dataclass
from chess_logic import board_to_tensor, reward_board

def grpo_chess_loss(
    logprobs_new: torch.Tensor,   # [G, T]  log πθ(a_{g,k,t} | s_{g,k,t})
    logprobs_old: torch.Tensor,   # [G, T]  log πold(a_{g,k,t} | s_{g,k,t})
    advantages: torch.Tensor,        # [G, T]
    clip_eps: float = 0.2,  # ε in the formula
    beta_kl: float = 0.0,   # β in the formula (0 = no explicit KL penalty)
    eps: float = 1e-8):

    # ------------------------------------------------------------
    # 3. Probability ratio r_{g,k,t}(θ)
    #
    #    r_{g,k,t}(θ) = πθ(a_{g,k,t}|s_{g,k,t}) / πold(a_{g,k,t}|s_{g,k,t})
    #                 = exp( logπθ - logπold )
    # ------------------------------------------------------------
    ratio = (logprobs_new - logprobs_old).exp() # [G, T]
    pg_unclipped = -advantages * ratio  # [G, T]
    pg_clipped = -advantages * ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) # [G, T]

    # Surrogate policy gradient loss (PPO-clip part)
    # This corresponds to the -E[min(...)] in the formula.
    policy_loss = torch.max(pg_unclipped, pg_clipped).mean()
    approx_kl = (logprobs_old - logprobs_new).mean()

    # KL penalty: β * E[ KL(...) ]
    kl_loss = beta_kl * approx_kl
    loss = policy_loss + kl_loss

    return loss, approx_kl


# Utils funcstions for GRPO
def group_advantage(group_rewards):
    mean_reward = group_rewards.mean(dim=-1, keepdim=True)
    std_reward = group_rewards.std(dim=-1, unbiased=False, keepdim=True) + 1e-8
    advantages = (group_rewards - mean_reward) / std_reward
    return advantages


def ppo_chess_loss(
    logprobs_new: torch.Tensor,   # [G, T]  log πθ(a_{g,k,t} | s_{g,k,t})
    logprobs_old: torch.Tensor,   # [G, T]  log πold(a_{g,k,t} | s_{g,k,t})
    advantages: torch.Tensor,        # [G, T]
    clip_eps: float = 0.2,  # ε in the formula
    pad_mask: torch.Tensor | None = None,  # [G, T], True = real, False = pad
    ):
    if pad_mask is None:
      pad_mask = torch.ones_like(logprobs_new, dtype=torch.bool)
    ratio = (logprobs_new - logprobs_old).exp() # [G, T]
    pg_unclipped = -advantages * ratio  # [G, T]
    pg_clipped = -advantages * ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) # [G, T]
    # Surrogate policy gradient loss (PPO-clip part)
    # This corresponds to the -E[min(...)] in the formula.
    policy_loss = torch.max(pg_unclipped, pg_clipped) * pad_mask.float()
    return policy_loss

def kl_penalty(logprobs_new: torch.Tensor, 
               logprobs_old: torch.Tensor,
               pad_mask=None):
    if pad_mask is None:
      pad_mask = torch.ones_like(logprobs_new, dtype=torch.bool)
    return (logprobs_old - logprobs_new)[pad_mask].mean()


def grpo_ppo_loss(
    logprobs_new: torch.Tensor,     # [G, T]
    logprobs_old: torch.Tensor,     # [G, T]
    group_rewards: torch.Tensor,    # [G,]
    pad_mask: torch.Tensor = None,  # [G, T]
    clip_ratio: float = 0.2,        # in paper this epsilon
    kl_coef: float = 0.01,          # in paper this is beta
    ) -> torch.Tensor:
    if logprobs_new.ndim == 2: # No batch input - unsqueeze
        logprobs_new = logprobs_new.unsqueeze(0)
        logprobs_old = logprobs_old.unsqueeze(0)
        group_rewards = group_rewards.unsqueeze(0)
        if pad_mask is not None:
            pad_mask = pad_mask.unsqueeze(0)

    if pad_mask is None:
        pad_mask = torch.ones_like(logprobs_new, dtype=torch.bool)

    B, G, T = logprobs_new.shape
    advantages_2d = group_advantage(group_rewards).detach()
    advantages = advantages_2d.unsqueeze(-1).expand(B, G, T)
    advantages = advantages * pad_mask.float()
    ppo_lose = ppo_chess_loss(logprobs_new,
                              logprobs_old,
                              advantages, 
                              clip_ratio,
                              pad_mask)
    valid_steps = pad_mask.sum().clamp_min(1)
    ppo_lose = ppo_lose.sum() / valid_steps
    kl_div = kl_penalty(logprobs_new, logprobs_old, pad_mask)
    loss = ppo_lose + kl_coef * kl_div
    return loss


# Trajectories sampling logic
@dataclass
class TrajectoriesSample:
    trajectories_log_probs: torch.Tensor # [G, T]
    trajectories_actinos: torch.Tensor   # [G, T]
    trajectories_states: torch.Tensor    # [G, T, SEQ]
    group_rewards: torch.Tensor          # [G,]
    pad_mask: torch.Tensor               # [G, T]

    @staticmethod
    def create_from_lists(trajectories_log_probs: list,
                          trajectories_actinos: list,
                          trajectories_states: list,
                          group_rewards: list,
                          pad_length: int,
                          device):
      # Pad
      pad_mask = torch.ones(len(trajectories_log_probs), pad_length, dtype=torch.bool)
      for g in range(len(trajectories_log_probs)):
        pad_length_g = pad_length - len(trajectories_log_probs[g])
        pad_mask[g, len(trajectories_log_probs[g]):] = False
        actions_tensor = torch.tensor(trajectories_actinos[g],
                                      dtype=torch.long)
        
        trajectories_log_probs[g] = F.pad(trajectories_log_probs[g], (0, pad_length_g))
        trajectories_actinos[g] = F.pad(actions_tensor, (0, pad_length_g))
        trajectories_states[g] = F.pad(trajectories_states[g], (0, 0, 0, pad_length_g))

      return TrajectoriesSample(torch.stack(trajectories_log_probs).to(device),
                                torch.stack(trajectories_actinos).to(device),
                                torch.stack(trajectories_states).to(device),
                                torch.tensor(group_rewards, 
                                             dtype=torch.float32,
                                             device=device),
                                pad_mask.to(device))


def sample_trajectories(model, num_trajectories, trajectory_depth, board):
    '''
    Sample trajectories from the current policy.
    Returns:
        trajectories_log_probs: [G, T]
        trajectories_actinos: [G, T]
        trajectories_states: [G, T, SEQ]
        group_rewards: [G,]
        pad_mask: [G, T] <-- in case of game end in trajectory
    
    '''
    device = model.device
    group_rewards = []
    trajectories_log_probs = []
    trajectories_actinos = []
    trajectories_states = []
    for _ in range(num_trajectories):
        env = board.copy()
        traj_log_probs = []
        traj_actions = []
        traj_states = []
        for step in range(trajectory_depth):
            if env.is_game_over(): break

            move, log_prob, idx = model.select_action(env)
            if move is None: break # Reach end of game - PAD!

            traj_log_probs.append(log_prob)
            traj_actions.append(idx)
            traj_states.append(board_to_tensor(env, model.device).squeeze(0))

            env.push(move)
        if len(traj_log_probs) == 0: continue # Skip if no moves

        group_rewards.append(reward_board(env, board))
        trajectories_log_probs.append(torch.stack(traj_log_probs))
        trajectories_actinos.append(traj_actions)
        trajectories_states.append(torch.stack(traj_states))

    return TrajectoriesSample.create_from_lists(trajectories_log_probs,
                                                trajectories_actinos,
                                                trajectories_states,
                                                group_rewards,
                                                trajectory_depth,
                                                model.device)