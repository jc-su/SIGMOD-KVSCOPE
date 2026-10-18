# Build the eKV eBPF collectors against the VENDORED libbpf (third_party/libbpf) -- not the
# distro libbpf 0.5 -- for func_name uprobe resolution and the generic SEC("uprobe").
# Requires: clang (14 or newer), libelf, and a kernel with BTF at /sys/kernel/btf/vmlinux.
# Kernel: Linux 5.8 or newer (BPF ring buffer). The programs use CO-RE, and vmlinux.h is generated
# from the running kernel's BTF with the vendored bpftool.
#
# Build the vendored libbpf+bpftool first (one-time) via scripts/build_ebpf.sh, then `make`.
CLANG   ?= clang
BPF_MCPU  ?= v3
BPFTOOL ?= $(abspath ../../third_party/bpftool/bpftool)
ARCH    := x86
CFLAGS  := -g -O2 -Wall
LIBBPF      := $(abspath ../../third_party/libbpf)
LIBBPF_INC  := $(LIBBPF)/usr/include
LIBBPF_A    := $(LIBBPF)/usr/lib64/libbpf.a
CLANG_BPF_SYS_INCLUDES ?= $(shell $(CLANG) -v -E - </dev/null 2>&1 \
	| sed -n '/<...> search starts here:/,/End of search list./{ s| \(/.*\)|-idirafter \1|p }')

all: kv_copy http_label ekv_readout kv_alloc storage_read

vmlinux.h:
	$(BPFTOOL) btf dump file /sys/kernel/btf/vmlinux format c > $@

# --- generic rules ---
%.bpf.o: %.bpf.c %.h vmlinux.h
	$(CLANG) -target bpf -mcpu=$(BPF_MCPU) -D__TARGET_ARCH_$(ARCH) $(CFLAGS) -I. -I$(LIBBPF_INC) \
		$(CLANG_BPF_SYS_INCLUDES) -c $< -o $@

%.skel.h: %.bpf.o
	$(BPFTOOL) gen skeleton $< > $@

kv_copy: kv_copy.c kv_copy.skel.h kv_copy.h
	$(CLANG) $(CFLAGS) -I. -I$(LIBBPF_INC) $< $(LIBBPF_A) -lelf -lz -o $@

# network-label collector: HTTP-request capture via recvfrom tracepoints.
http_label: http_label.c http_label.skel.h http_label.h ringbuf_report.h
	$(CLANG) $(CFLAGS) -I. -I$(LIBBPF_INC) $< $(LIBBPF_A) -lelf -lz -o $@

# zero-touch readout: uprobe-triggered bpf_probe_read_user of the pass's published
# host_va -> {slots, cycles}, the eBPF realization of the /proc read in zero_touch.py.
ekv_readout: ekv_readout.c ekv_readout.skel.h ekv_readout.h ringbuf_report.h
	$(CLANG) $(CFLAGS) -I. -I$(LIBBPF_INC) $< $(LIBBPF_A) -lelf -lz -o $@

# KV-pool allocation capture: cudaMalloc base+size -> from_alloc (zero-touch addr->block).
kv_alloc: kv_alloc.c kv_alloc.skel.h kv_alloc.h ringbuf_report.h
	$(CLANG) $(CFLAGS) -I. -I$(LIBBPF_INC) $< $(LIBBPF_A) -lelf -lz -o $@

# storage-tier read capture: times read()/pread64() (size-filtered = KV chunks) so a
# lower-tier (disk/network) fetch is joined to the following H2D -> full miss cost.
storage_read: storage_read.c storage_read.skel.h storage_read.h ringbuf_report.h
	$(CLANG) $(CFLAGS) -I. -I$(LIBBPF_INC) $< $(LIBBPF_A) -lelf -lz -o $@

clean:
	rm -f kv_copy http_label ekv_readout kv_alloc storage_read *.bpf.o *.skel.h vmlinux.h

.PHONY: all clean
