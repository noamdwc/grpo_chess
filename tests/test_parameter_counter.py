import torch.nn as nn

from src.parameter_counter import ParameterCounter


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4, 3, bias=True)   # 12 + 3 = 15
        self.l2 = nn.Linear(3, 2, bias=False)  # 6
        self.l2.weight.requires_grad = False

    def forward(self, x):
        return self.l2(self.l1(x))


def test_parameter_counter_counts_total_trainable_and_frozen() -> None:
    wrapped = ParameterCounter(_TinyNet())
    summary = wrapped.summary()

    assert wrapped.count_parameters() == 21
    assert wrapped.count_parameters(trainable_only=True) == 15
    assert summary.total == 21
    assert summary.trainable == 15
    assert summary.frozen == 6
