import json
from pathlib import Path

import torch

from src.distill.convert_deepmind_data import ConvertConfig, convert


def test_convert_writes_conversion_stats(tmp_path, monkeypatch):
    output_dir = tmp_path / "distill_data"

    monkeypatch.setattr(
        "src.distill.convert_deepmind_data.download_shard",
        lambda shard_idx, download_dir: str(tmp_path / f"dummy_{shard_idx}.bag"),
    )
    monkeypatch.setattr("src.distill.convert_deepmind_data.count_bag_records", lambda _path: 4)
    monkeypatch.setattr(
        "src.distill.convert_deepmind_data.read_bag_records",
        lambda _path: [b"a", b"b", b"c", b"d"],
    )

    decoded = {
        b"a": ("fen_a", "m1", 0.90),
        b"b": ("fen_a", "m2", 0.80),
        b"c": ("fen_b", "m3", 0.85),
        b"d": ("fen_c", "m4", 0.40),  # filtered by min_win_prob
    }
    monkeypatch.setattr(
        "src.distill.convert_deepmind_data.decode_action_value",
        lambda raw: decoded[raw],
    )

    def fake_make_grouped_sample(fen, moves_and_probs, top_k, temperature):
        if fen == "fen_b":
            return None
        assert top_k == 8
        assert temperature == 1.0
        return {
            "board_tokens": torch.zeros(77, dtype=torch.long),
            "legal_mask": torch.zeros(1968, dtype=torch.bool),
            "teacher_action_indices": torch.tensor([1, 2], dtype=torch.long),
            "teacher_probs": torch.tensor([0.6, 0.4], dtype=torch.float32),
        }

    monkeypatch.setattr(
        "src.distill.convert_deepmind_data.make_grouped_sample",
        fake_make_grouped_sample,
    )

    saved_shards: list[tuple[Path, int]] = []

    def fake_save_shard(samples, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"ok")
        saved_shards.append((path, len(samples)))

    monkeypatch.setattr("src.distill.convert_deepmind_data.save_shard", fake_save_shard)

    cfg = ConvertConfig(
        num_shards=1,
        min_win_prob=0.55,
        output_dir=str(output_dir),
        shard_size=10,
    )
    convert(cfg)

    assert saved_shards == [(output_dir / "shard_0000.pt", 1)]

    stats_path = output_dir / "conversion_stats.json"
    assert stats_path.exists()
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    assert stats["records_total"] == 4
    assert stats["records_below_min_win_prob"] == 1
    assert stats["positions_grouped"] == 2
    assert stats["positions_kept"] == 1
    assert stats["positions_skipped"] == 1
    assert stats["samples_saved"] == 1
    assert stats["output_shards"] == 1
    assert stats["top_k_histogram"] == {"2": 1}
    assert stats["teacher_moves_mean_per_kept_position"] == 2.0
