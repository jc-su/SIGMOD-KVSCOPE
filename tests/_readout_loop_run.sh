#!/usr/bin/env bash
# RETIRED (2026-07-02, device-timer change): this gate validated the pure-eBPF sync-anchored
# cycles readout loop (ekv_readout streaming timer snapshots; 499/499 clean windows). The timer
# accumulator then moved to DEVICE memory to kill the ~24% host-mapped per-CTA PCIe atomic storm
# (commit 78ad5c9), and eBPF cannot read device memory -- so the byte-readout role moved to the
# out-of-process daemon (observe/daemon.py), which opens the pass's exported CUDA-IPC handles
# (timer_ipc / attn_weight_ipc) and D2Hs from its own context; see tests/_daemon_run.sh for the live
# cross-process gate. eBPF keeps the roles it is right for: MOVEMENT observation and sync-anchor
# timing. The capture-buffer half of ekv_readout (host-mapped, /proc-readable) remains valid.
echo "RETIRED: pure-eBPF cycles readout (timer is device-resident since 78ad5c9)."
echo "         Byte readout -> observe/daemon.py over CUDA IPC (gate: tests/_daemon_run.sh);"
echo "         eBPF keeps movement observation + sync-anchor timing."
exit 0
