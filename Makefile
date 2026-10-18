# eKV top-level convenience targets.
#   make test   -- CPU-pure test gate (no GPU/sudo); exits non-zero on any failure
#   make ebpf   -- build the eBPF collectors (needs the vendored toolchain; see scripts/build_ebpf.sh)
#   make clean  -- remove eBPF build artifacts
.PHONY: test ebpf clean

test:
	bash tests/run_cpu.sh

ebpf:
	$(MAKE) -C observe/ebpf

clean:
	$(MAKE) -C observe/ebpf clean
