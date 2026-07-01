# WRA-RR: Winnowing Representative Anchor Redundancy Reduction

WRA-RR is a lightweight replacement candidate for MalSnif's prefix-tree process event sequence reduction. It does not build a prefix tree, split loop bodies, match historical phrases, add new tokens, or attach extra risk rules. It selects local representative anchors with one fixed-size sliding window.

## Definition

For a process event sequence:

\[
S_v=[x_1,x_2,\ldots,x_n]
\]

WRA-RR computes a deterministic event-position hash:

\[
h_i = H(\kappa(x_i), i)
\]

where \(\kappa(x_i)\) is the sanitized event-token key and \(i\) is the event position. The position term avoids pathological ties in long identical runs while preserving deterministic preprocessing.

For every window:

\[
W_t=[t,t+1,\ldots,t+w-1]
\]

WRA-RR selects the rightmost minimum-hash event:

\[
a_t = \max \{ i \in W_t \mid h_i = \min_{j\in W_t} h_j \}
\]

The reduced sequence is obtained by sorting and deduplicating all selected anchors while always retaining the first and last events:

\[
S'_v=[x_i\mid i\in sort(unique(\{1,n\}\cup\{a_t\}))]
\]

## Default parameter

```text
w = 11
```

With random-like hashes, winnowing has expected anchor density close to:

\[
D(w) \approx \frac{2}{w+1}
\]

so the expected compression ratio is:

\[
CR(w) \approx 1-\frac{2}{w+1}
\]

For \(w=11\), the expected compression ratio is about 83.3%, close to the previously observed prefix_tree reduction level.

## Experiment

Run:

```bash
DEVICE=0 bash scripts/run_cadets_wra_rr_verdict.sh
```

The script compares only:

```text
off
prefix_tree
winnowing_anchor
```

on E1_eha_only with paired seeds and writes a single upload folder:

```text
runs/cadets_wra_rr_verdict_<timestamp>_autostop_win<WINDOW_EVENTS>/analysis_bundle/collected/
```
