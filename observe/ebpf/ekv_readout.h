/* Shared record layout for the eKV zero-touch readout (ekv_readout).
 *
 * One snapshot of a GPU-pass-published buffer, lifted out of the engine process via
 * bpf_probe_read_user — the eBPF realization of compiler-capture readout (the /proc
 * variant lives in observe/zero_touch.py). EKV_BUF_CAP is a power of two and MUST
 * exceed the published buffer size (max_rows*ncap*4): the read is bounded by
 * `& (EKV_BUF_CAP-1)` so the variable-size probe_read verifies on 5.15 (the same
 * idiom http_label/kv_copy use), which preserves the byte count exactly only while
 * the buffer is strictly smaller than the cap. The served default capture buffer is
 * 32 KiB (max_rows=64, ncap=128); 64 KiB covers it. Bump this for larger buffers. */
#ifndef EKV_READOUT_H
#define EKV_READOUT_H

#define EKV_BUF_CAP 65536

struct ekv_buf_rec {
  unsigned int kind;            /* 0 = capture (block_table rows) | 1 = timer (cycles) */
  unsigned int len;             /* bytes captured (< EKV_BUF_CAP) */
  unsigned long long ts_ns;     /* bpf_ktime_get_ns() */
  unsigned char data[EKV_BUF_CAP];
};

#endif /* EKV_READOUT_H */
