/* Shared record for the eKV storage-tier read probe (storage_read).
 *
 * The KV miss-cost picture has a gap: when a tiered backend (e.g. LMCache) serves a
 * block from a LOWER tier (local disk, remote), it first READS it into a host buffer
 * (a file read / network recv) and THEN issues the host->device copy that kv_copy
 * times. The cudaMemcpy data plane sees only the H2D; the tier read that precedes it
 * is invisible -- so a disk-tier miss looks as cheap as a DRAM-tier one. This probe
 * times the read syscall (PID- and size-filtered: large reads are KV chunks), so the
 * analysis can join it to the following H2D on the same thread and report the FULL
 * tier miss cost = tier_read + H2D, turning an unobserved component into a measured one. */
#ifndef EKV_STORAGE_READ_H
#define EKV_STORAGE_READ_H

struct read_rec {
  unsigned long long ts_ns;    /* completion (uretprobe) timestamp */
  unsigned int pid;
  unsigned int tid;            /* join key to the following H2D on the same thread */
  unsigned long long fd;
  unsigned long long bytes;    /* bytes actually read (return value) */
  unsigned long long dur_ns;   /* read syscall duration = the tier-read latency */
};

#endif /* EKV_STORAGE_READ_H */
