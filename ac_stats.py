import copy
from typing import Callable, Tuple

import torch

from torch.distributed._tools.sac_estimator import SACEstimator
from test_model import GPT, GPTConfig, loss_fn
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode


def collect_ac_tradeoff_stats(
    model: nn.Module,
    inp_and_target: Tuple[torch.Tensor, torch.Tensor],
    loss_fn: Callable = lambda x, y: sum(x, y),
):
    inp, target = inp_and_target
    sace = SACEstimator()
    with sace('operator-level-benchmark'):
        loss = loss_fn(model(inp), target)
    loss.backward()
    sace.pwlf_sac_tradeoff_curve(n_segments=2, save_tradeoff_graphs=True)
    sace.display_modulewise_sac_stats(depth=4, print_tabular=True)
    return copy.deepcopy(sace.sac_mod_tradeoff_stats)


if __name__ == "__main__":

    with FakeTensorMode():
        dev = torch.device(torch.cuda.current_device())
        n_layer = 6
        vocab_size = 8192
        config = GPTConfig(
            block_size=8192,
            n_layer=n_layer,
            dropout=0.01,
            vocab_size=vocab_size,
            checkpoint_activations=False,
        )
        with torch.device(dev):
            model = GPT(config)
        torch.manual_seed(1)
        bsz, seq_len = 16, 8192
        src = torch.randint(0, vocab_size, (bsz, seq_len), device=dev)
        tgt = torch.randint(0, vocab_size, (bsz, seq_len), device=dev)
        inp = (src, tgt)
        collect_ac_tradeoff_stats(model, inp, loss_fn)
