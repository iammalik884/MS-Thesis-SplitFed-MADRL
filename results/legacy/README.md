# LEGACY results — pre-neighbour architecture (do not compare directly)

`step5_fcfs_baseline_seed42.json` is the original Step 5 FCFS result, kept
**byte-for-byte unchanged** as historical evidence (moved here from
`results/` by the V2 audit fixes). It was produced by commit `eb60733`
(backup tag `v0.5-pre-audit-fixes`).

It is **not comparable** with the V2 baselines in `results/baselines/` because:

| Aspect | Legacy (this file) | V2 |
|---|---|---|
| Local tier | tiny bedside node (capacity 1.0) per department | the department's OWN edge node |
| Edge tier | the department's own node | a NEIGHBOUR department's node (+1 step transfer) |
| Routing rule | compute_demand thresholds only | demand → Cloud, else source backlog → Local / least-loaded neighbour |
| Same-step order | generator loop order (Critical/Emergency first) | seeded shuffle |
| Throughput | completed / (20 + arbitrary 10 drain) steps | window completions / 20 window steps |
| Deadline rate | completed tasks only | on-time / late / overdue-unfinished / not-yet-due |
| Provenance | seed + absolute config path only | full config snapshot, git commit + dirty flag, versions |

ROMAN URDU: Yeh purana result sirf record ke liye hai. Is ke numbers V2 ke
numbers se seedha compare na karein — architecture aur metric definitions
dono badal chuke hain.
