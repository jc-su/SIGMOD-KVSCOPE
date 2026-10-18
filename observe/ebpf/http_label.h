/* Shared record layout for the eKV network-label collector (http_label).
 *
 * Captures the HOST-side bytes a served engine recv()s on its API socket — the
 * inbound HTTP request — so the request identity can be derived from the NETWORK
 * (prompt -> tokens) and matched to the engine's input_ids, with NO engine hook.
 * Plain C scalars so it compiles unchanged on both sides (LP64). */
#ifndef EKV_HTTP_LABEL_H
#define EKV_HTTP_LABEL_H

/* Bytes of one recv buffer captured. MUST be a power of two: the exit probe bounds
 * the variable read with `& (HTTP_CAP-1)` (a u64 `<=` compare leaves the low bits
 * var_off messy and the 5.15 verifier rejects the variable-size probe_read — same
 * constraint kv_copy's on_meta hit). 8192 covers a small completion request
 * (headers + short JSON body); larger prompts arrive across several recvs and the
 * userspace side reassembles per-connection. */
#define HTTP_CAP 8192

struct http_rec {
  unsigned long long ts_ns;   /* bpf_ktime_get_ns() == Python time.monotonic_ns() */
  unsigned int pid;
  unsigned int tid;
  long long fd;               /* per-connection reassembly key: the socket fd for the syscall path,
                                 or the SSL* pointer for the TLS path. 64-bit so a TLS connection id
                                 is not truncated (a 32-bit field could merge two SSL* that share
                                 their low 32 bits). */
  unsigned int len;           /* bytes captured (<= HTTP_CAP) */
  unsigned char data[HTTP_CAP];
};

#endif /* EKV_HTTP_LABEL_H */
