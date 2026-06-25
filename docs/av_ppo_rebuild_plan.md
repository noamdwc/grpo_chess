# AV PPO Rebuild Plan

Branch: `av-ppo-rebuild`

Goal: rebuild the main post-training path around a searchless PPO setup that can show RL improvement over the pretrained model before reintroducing reasoning/search.

Plan:

1. Remove reasoning/search from the active training path.
2. Fix the training loop so it is fully PPO: sampled action, post-action gain, critic value, advantage `gain - value`, clipped PPO loss, value loss, entropy, and KL regularization.
3. Adapt the strong DeepMind action-value model:
   - actor: AV backbone plus next-action policy head
   - critic: AV backbone plus scalar state-value head
4. Use Stockfish only as scalar feedback for the post-action state.
5. Run experiments only after the PPO path is clean and tested.
6. Merge successful PPO/AV work back to `search_refactor`.
7. Add reasoning/thinking later, then only after that prepare a merge to `main`.

Invariant: this remains a searchless chess project. Do not add MCTS, tree search, or related search-based methods.
