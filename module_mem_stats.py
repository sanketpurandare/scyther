import copy
from contextlib import nullcontext
from typing import cast

import torch
from torch.distributed._tools.mem_tracker import MemTracker, _ModState, _ModMemStats, _MemRefType
from torch._subclasses.fake_tensor import FakeTensorMode

from test_model import GPT, GPTConfig

def collect_mem_stats(print_stats: bool = False):
    dev = torch.device(torch.cuda.current_device())
    n_layer = 6
    vocab_size = 8192
    config = GPTConfig(
        block_size=512, n_layer=n_layer, vocab_size=vocab_size
    )
    with torch.device(dev):
        model = GPT(config)
    optim = torch.optim.Adam(model.parameters(), lr=1e-2, foreach=True)
    torch.manual_seed(1)
    bsz, seq_len = 64, 512

    def train_step():
        src = torch.randint(0, vocab_size, (bsz, seq_len), device=dev)
        tgt = torch.randint(0, vocab_size, (bsz, seq_len), device=dev)
        inp = (src, tgt)
        loss = model(*inp).sum()
        loss.backward()
        optim.step()
        optim.zero_grad()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()

    mem_tracker = MemTracker()
    mem_tracker.track_external(model, optim)
    with mem_tracker as mt:
        for iter_idx in range(2):
            train_step()
            if iter_idx == 0:
                mt.reset_mod_stats()

    mt.display_modulewise_snapshots(depth=4, units="MB", tabulate=True)
    mt.display_snapshot("peak", units="MB", tabulate=True)
    tracker_max = mt.get_tracker_snapshot('peak')[dev]['Total']
    cuda_max = torch.cuda.max_memory_allocated()
    accuracy = tracker_max / (cuda_max + 1)  # +1 to avoid div by 0
    print(f"Tracker Max: {tracker_max}, CUDA Max: {cuda_max}, Accuracy: {accuracy}")
    print(accuracy >= 0.9)
    module_mem_stats = copy.deepcopy(mt.memory_tracking)
    if print_stats:
        for mod in model.modules():
            mod_stat = module_mem_stats.get(mod, None)
            if mod_stat:
                mod_stat = cast(_ModMemStats, mod_stat)
                print(mod_stat.mod_fqn)
                # Access using state, #call, device, type
                print(f"Fw Peak Activation: {mod_stat.snapshots[_ModState.PEAK_FW][-1][dev][_MemRefType.ACT]}")
                print(f"Bw Peak Activation: {mod_stat.snapshots[_ModState.PEAK_BW][-1][dev][_MemRefType.ACT]}")
                print(f"Bw Peak Temp (Act Grads): {mod_stat.snapshots[_ModState.PEAK_BW][-1][dev][_MemRefType.TEMP]}")

if __name__ == "__main__":
    use_fake_mode = False
    with FakeTensorMode() if use_fake_mode else nullcontext():
        collect_mem_stats(print_stats=True)
