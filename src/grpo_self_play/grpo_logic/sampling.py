import random
import chess
import torch
import torch.nn.functional as F
from dataclasses import dataclass

from src.grpo_self_play.searchless_chess_imports import ACTION_TO_MOVE, SEQUENCE_LENGTH
from src.grpo_self_play.chess.chess_logic import board_to_tensor, reward_board, get_legal_moves_indices 


# Trajectories sampling logic
@dataclass
class TrajectoriesSample:
    trajectories_log_probs: torch.Tensor # [B, G, T]
    trajectories_actinos: torch.Tensor   # [B, G, T]
    trajectories_states: torch.Tensor    # [B, G, T, SEQ]
    group_rewards: torch.Tensor          # [B, G]
    pad_mask: torch.Tensor               # [B, G, T]


def batched_policy_step(model, boards, temperature: float = 1.0):
  N = len(boards)
  if N == 0: return None
  device = next(model.parameters()).device
  states_list = []
  legal_indices = []
  for board in boards:
    state = board_to_tensor(board, device=device)
    states_list.append(state)
    legal_indices.append(get_legal_moves_indices(board))
  states_tensor = torch.cat(states_list, dim=0) # [N, SEQ]
  logits = model(states_tensor) # [N, O]
  masked_logits = torch.full_like(logits, -float("inf"))
  for i in range(N):
    if len(legal_indices[i]) == 0: continue
    masked_logits[i, legal_indices[i]] = logits[i, legal_indices[i]] / temperature
  probs = F.softmax(masked_logits, dim=-1)
  action_idx = torch.multinomial(probs, 1).squeeze(1) # [N,]
  chosen_probs = probs.gather(1, action_idx.unsqueeze(1)).squeeze(1) # [N,]
  chosen_log_probs = torch.log(chosen_probs) # [N,]

  moves = []
  for i, idx in enumerate(action_idx.tolist()):
      uci = ACTION_TO_MOVE[idx]
      move = chess.Move.from_uci(uci)
      if move not in boards[i].legal_moves:
          legal_moves = list(boards[i].legal_moves)
          move = random.choice(legal_moves) if legal_moves else None
      moves.append(move)
  return action_idx, chosen_log_probs, moves, states_tensor


def sample_trajectories_batched(model, boards, num_trajectories, trajectory_depth):
  device = next(model.parameters()).device
  B, G, T = len(boards), num_trajectories, trajectory_depth
  if B == 0: return None

  envs = [boards[b].copy() for b in range(B) for _ in range(G)] # Lenght of B*G
  # Per (b, g) storage as nested lists
  traj_log_probs = [[[] for _ in range(G)] for _ in range(B)]
  traj_actions = [[[] for _ in range(G)] for _ in range(B)]
  traj_states = [[[] for _ in range(G)] for _ in range(B)]

  # Rollout (sample trajectoris at batches)
  for _ in range(T):
    active_env_idx = [i for i, e in enumerate(envs) if not e.is_game_over()]
    if not active_env_idx: break

    active_boards = [envs[i] for i in active_env_idx]
    roll_out_step = batched_policy_step(model, active_boards, temperature=1.0)
    if roll_out_step is None: break

    action_indices, log_probs, moves, states_batch = roll_out_step
    if action_indices is None: break

    for j, env_idx_j in enumerate(active_env_idx):
      move_j = moves[j]
      if move_j is None: continue # End of game for this env
      b_idx = env_idx_j // G
      g_idx = env_idx_j % G
      state_j = states_batch[j]
      traj_log_probs[b_idx][g_idx].append(log_probs[j])
      traj_actions[b_idx][g_idx].append(int(action_indices[j].item()))
      traj_states[b_idx][g_idx].append(state_j)
      envs[env_idx_j].push(move_j)

  # Reward per final state
  group_rewards = torch.zeros(B, G, dtype=torch.float32, device=device)
  for env_idx, env in enumerate(envs):
    b_idx = env_idx // G
    g_idx = env_idx % G
    group_rewards[b_idx, g_idx] = reward_board(env, boards[b_idx])

  # Allocate padded tensors
  trajectories_log_probs = torch.zeros(B, G, T, dtype=torch.float32, device=device)
  trajectories_actions = torch.zeros(B, G, T, dtype=torch.long,   device=device)
  trajectories_states = torch.zeros(B, G, T, SEQUENCE_LENGTH, dtype=torch.long, device=device)
  pad_mask = torch.zeros(B, G, T, dtype=torch.bool,  device=device)
  for b in range(B):
    for g in range(G):
      L = len(traj_log_probs[b][g])
      assert L <= T, f"Trajectory length {L} exceeds pad_length {T}"
      pad_mask[b, g, :L] = True
      trajectories_log_probs[b, g, :L] = torch.stack(traj_log_probs[b][g], dim=0)
      trajectories_actions[b, g, :L] = torch.tensor(traj_actions[b][g], dtype=torch.long, device=device)
      trajectories_states[b, g, :L] = torch.stack(traj_states[b][g], dim=0)

  return TrajectoriesSample(trajectories_log_probs,
                            trajectories_actions,
                            trajectories_states,
                            group_rewards,
                            pad_mask)
                            