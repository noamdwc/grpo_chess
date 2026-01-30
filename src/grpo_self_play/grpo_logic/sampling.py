import os
import random
from typing import List, Optional
import chess
import chess.engine
import torch
import torch.nn.functional as F
from dataclasses import dataclass

from src.grpo_self_play.chess.rewards import reward_board, evaluate_board, normalize_cp
from src.grpo_self_play.models import ChessTransformer
from src.grpo_self_play.searchless_chess_imports import ACTION_TO_MOVE, SEQUENCE_LENGTH, MOVE_TO_ACTION
from src.grpo_self_play.chess.chess_logic import board_to_tensor,  get_legal_moves_mask
from src.grpo_self_play.chess.stockfish import StockfishManager, StockfishConfig


# Process-safe Stockfish engine for teacher forcing
_teacher_engine: chess.engine.SimpleEngine | None = None
_teacher_engine_pid: int | None = None


def get_teacher_engine(cfg: StockfishConfig | None = None) -> chess.engine.SimpleEngine:
    """Get a process-safe Stockfish engine for teacher forcing."""
    global _teacher_engine, _teacher_engine_pid
    pid = os.getpid()
    if pid != _teacher_engine_pid:
        _teacher_engine = None
        _teacher_engine_pid = pid
    if _teacher_engine is None:
        _teacher_engine = StockfishManager.get_engine(f"teacher_forcing_{pid}", cfg)
    return _teacher_engine


def get_stockfish_move(board: chess.Board, depth: int = 4) -> Optional[chess.Move]:
    """Get the best move from Stockfish for a given board position.

    Args:
        board: Chess board position
        depth: Stockfish search depth

    Returns:
        Best move from Stockfish, or None if no move available
    """
    if board.is_game_over():
        return None

    engine = get_teacher_engine()
    limit = chess.engine.Limit(depth=depth)
    result = engine.play(board, limit)
    return result.move


# Trajectories sampling logic
@dataclass
class TrajectoriesSample:
    """Container for batched trajectory samples.

    Attributes:
        trajectories_log_probs: Log probabilities of sampled actions [B, G, T]
        trajectories_actions: Action indices [B, G, T]
        trajectories_states: State tensors [B, G, T, SEQ]
        group_rewards: Final rewards for each trajectory group [B, G] (for logging)
        step_rewards: Per-step rewards [B, G, T] where step_rewards[b,g,t] = eval(s_{t+1}) - eval(s_t)
        pad_mask: Mask indicating valid steps, True=valid, False=padding [B, G, T]
        trajectories_legal_masks: Legal moves masks [B, G, T, A]
        raw_step_cp: Raw centipawn step rewards [B, G, T] (for logging, not normalized)
    """
    trajectories_log_probs: torch.Tensor  # [B, G, T]
    trajectories_actions: torch.Tensor    # [B, G, T]
    trajectories_states: torch.Tensor     # [B, G, T, SEQ]
    group_rewards: torch.Tensor           # [B, G]
    step_rewards: torch.Tensor            # [B, G, T]
    pad_mask: torch.Tensor                # [B, G, T]
    trajectories_legal_masks: torch.Tensor  # [B, G, T, A]
    raw_step_cp: torch.Tensor             # [B, G, T] - raw centipawn differences


def batched_policy_step(model: ChessTransformer, boards: List[chess.Board], temperature: float = 1.0) -> Optional[tuple]:
    """Sample actions from policy for a batch of boards.
    
    Args:
        model: Chess transformer model
        boards: List of chess board positions
        temperature: Temperature for sampling
        
    Returns:
        Tuple of (action_indices, log_probs, moves, states_tensor, legal_mask) or None if empty
    """
    N = len(boards)
    if N == 0:
        return None
    device = next(model.parameters()).device
    states_list = []
    legal_masks = []
    for board in boards:
        state = board_to_tensor(board, device=device)
        states_list.append(state)
        mask = get_legal_moves_mask(board, device=device)
        if mask.ndim == 2:
            mask = mask.squeeze(0)
        assert mask.ndim == 1, f"legal_moves_mask must be 1D [A], got {mask.shape}"
        legal_masks.append(mask)

    states_tensor = torch.cat(states_list, dim=0)  # [N, SEQ]
    legal_mask = torch.stack(legal_masks, dim=0)     # [N, A] bool
    assert legal_mask.dtype == torch.bool, "legal_mask must be bool dtype"
    assert legal_mask.shape[0] == N, f"legal_mask batch size mismatch {legal_mask.shape[0]} vs {N}"
    assert legal_mask.shape[1] == model.action_size, f"legal_mask action size mismatch {legal_mask.shape[1]} vs {model.action_size}"
    if not legal_mask.any(dim=1).all():
        bad = (~legal_mask.any(dim=1)).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"Empty legal mask for boards: {bad}")
    probs = model.get_legal_moves_probs(states_tensor, legal_mask, temperature)  # [N, O]

    action_idx = torch.multinomial(probs, 1).squeeze(1)  # [N,]
    chosen_probs = probs.gather(1, action_idx.unsqueeze(1)).squeeze(1)  # [N,]
    chosen_log_probs = torch.log(chosen_probs + 1e-12)  # [N,], avoid log(0)

    # Convert action indices to moves, ensure legality
    moves = []
    for i, idx in enumerate(action_idx.tolist()):
        uci = ACTION_TO_MOVE[idx]
        move = chess.Move.from_uci(uci)
        if move not in boards[i].legal_moves:
            raise ValueError(f"Sampled illegal move {uci} for board:\n{boards[i]}")
        moves.append(move)
    return action_idx, chosen_log_probs, moves, states_tensor, legal_mask


def sample_trajectories_batched(model: ChessTransformer,
                                boards: List[chess.Board],
                                num_trajectories: int,
                                trajectory_depth: int,
                                reward_depth: int = 4,
                                temperature: float = 1.0,
                                teacher_forcing_prob: float = 0.0,
                                teacher_forcing_depth: int = 4) -> Optional[TrajectoriesSample]:
    """Sample multiple trajectories from each board position using the policy model.

    Args:
        model: Chess transformer model for action selection
        boards: List of starting board positions [B]
        num_trajectories: Number of trajectory groups per board (G)
        trajectory_depth: Maximum depth of each trajectory (T)
        reward_depth: Stockfish depth for reward computation (default: 4)
        temperature: Temperature for action sampling (default: 1.0, >1 increases exploration)
        teacher_forcing_prob: Probability of using Stockfish for rival moves (default: 0.0)
        teacher_forcing_depth: Stockfish depth for teacher forcing moves (default: 4)

    Returns:
        TrajectoriesSample containing batched trajectory data, or None if no boards
    """
    device = next(model.parameters()).device
    B, G, T = len(boards), num_trajectories, trajectory_depth
    if B == 0:
        return None

    # Create B*G copies of boards for parallel trajectory sampling
    envs = [boards[b].copy() for b in range(B) for _ in range(G)]  # Length of B*G
    # Per (b, g) storage as nested lists
    traj_log_probs = [[[] for _ in range(G)] for _ in range(B)]
    traj_actions = [[[] for _ in range(G)] for _ in range(B)]
    traj_states = [[[] for _ in range(G)] for _ in range(B)]
    traj_legal_masks = [[[] for _ in range(G)] for _ in range(B)]
    traj_step_rewards = [[[] for _ in range(G)] for _ in range(B)]
    traj_raw_step_cp = [[[] for _ in range(G)] for _ in range(B)]  # Raw centipawn differences for logging

    # Track POV and previous raw eval for each trajectory (we normalize step rewards later)
    pov_is_white = [(boards[b].turn == chess.WHITE) for b in range(B) for _ in range(G)]
    prev_evals_raw = [evaluate_board(boards[b], pov_is_white[b * G], depth=reward_depth, normalize=False)
                      for b in range(B) for _ in range(G)]

    # Rollout: sample trajectories in batches
    for t in range(T):
        active_env_idx = [i for i, e in enumerate(envs) if not e.is_game_over()]
        if not active_env_idx:
            break

        # Determine if this is the rival's turn (odd timesteps)
        is_rival_turn = (t % 2 == 1)
        use_teacher_forcing = is_rival_turn and teacher_forcing_prob > 0 and random.random() < teacher_forcing_prob

        active_boards = [envs[i] for i in active_env_idx]
        roll_out_step = batched_policy_step(model, active_boards, temperature=temperature)
        if roll_out_step is None:
            break

        action_indices, log_probs, moves, states_batch, legal_mask = roll_out_step
        if action_indices is None:
            break

        for j, env_idx_j in enumerate(active_env_idx):
            move_j = moves[j]
            if move_j is None:
                continue  # End of game for this env
            b_idx = env_idx_j // G
            g_idx = env_idx_j % G
            state_j = states_batch[j]

            # Teacher forcing: override rival's move with Stockfish
            if use_teacher_forcing:
                sf_move = get_stockfish_move(envs[env_idx_j], depth=teacher_forcing_depth)
                if sf_move is not None and sf_move in envs[env_idx_j].legal_moves:
                    move_j = sf_move
                    # Update action index to match the Stockfish move
                    action_indices[j] = MOVE_TO_ACTION[move_j.uci()]

            traj_log_probs[b_idx][g_idx].append(log_probs[j])
            traj_actions[b_idx][g_idx].append(int(action_indices[j].item()))
            traj_states[b_idx][g_idx].append(state_j)
            traj_legal_masks[b_idx][g_idx].append(legal_mask[j])
            envs[env_idx_j].push(move_j)

            # Compute step reward: eval(new_state) - eval(prev_state)
            # Get raw centipawn value, then normalize for step_rewards
            new_eval_raw = evaluate_board(envs[env_idx_j], pov_is_white[env_idx_j], depth=reward_depth, normalize=False)
            raw_step_cp = new_eval_raw - prev_evals_raw[env_idx_j]
            step_reward = normalize_cp(new_eval_raw) - normalize_cp(prev_evals_raw[env_idx_j])
            traj_step_rewards[b_idx][g_idx].append(step_reward)
            traj_raw_step_cp[b_idx][g_idx].append(raw_step_cp)
            prev_evals_raw[env_idx_j] = new_eval_raw

    # Compute group_rewards for logging (sum of step rewards = final - initial)
    group_rewards = torch.zeros(B, G, dtype=torch.float32, device=device)
    for env_idx, env in enumerate(envs):
        b_idx = env_idx // G
        g_idx = env_idx % G
        group_rewards[b_idx, g_idx] = reward_board(env, boards[b_idx], depth=reward_depth, movetime_ms=0)

    # Allocate padded tensors
    trajectories_log_probs = torch.zeros(B, G, T, dtype=torch.float32, device=device)
    trajectories_actions = torch.zeros(B, G, T, dtype=torch.long, device=device)
    trajectories_states = torch.zeros(B, G, T, SEQUENCE_LENGTH, dtype=torch.long, device=device)
    trajectories_legal_masks = torch.zeros(B, G, T, model.action_size, dtype=torch.bool, device=device)
    trajectories_legal_masks[..., 0] = True  # Ensure at least one legal move (to avoid empty legal masks -> NaNs in log_softmax)
    step_rewards = torch.zeros(B, G, T, dtype=torch.float32, device=device)
    raw_step_cp = torch.zeros(B, G, T, dtype=torch.float32, device=device)
    pad_mask = torch.zeros(B, G, T, dtype=torch.bool, device=device)
    for b in range(B):
        for g in range(G):
            L = len(traj_log_probs[b][g])
            assert L <= T, f"Trajectory length {L} exceeds pad_length {T}"
            pad_mask[b, g, :L] = True
            trajectories_log_probs[b, g, :L] = torch.stack(traj_log_probs[b][g], dim=0)
            trajectories_actions[b, g, :L] = torch.tensor(traj_actions[b][g], dtype=torch.long, device=device)
            trajectories_states[b, g, :L] = torch.stack(traj_states[b][g], dim=0)
            if L > 0:
                trajectories_legal_masks[b, g, :L] = torch.stack(traj_legal_masks[b][g], dim=0)
                step_rewards[b, g, :L] = torch.tensor(traj_step_rewards[b][g], dtype=torch.float32, device=device)
                raw_step_cp[b, g, :L] = torch.tensor(traj_raw_step_cp[b][g], dtype=torch.float32, device=device)

    return TrajectoriesSample(trajectories_log_probs,
                              trajectories_actions,
                              trajectories_states,
                              group_rewards,
                              step_rewards,
                              pad_mask,
                              trajectories_legal_masks,
                              raw_step_cp)
                            