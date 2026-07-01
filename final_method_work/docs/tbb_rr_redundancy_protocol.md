# TBB-RR redundancy protocol

**TBB-RR** means **Target-Budget Boundary Redundancy Reduction**.

It is a minimal replacement candidate for MalSnif's prefix-tree sequence reduction.  Instead of identifying loop bodies or constructing a prefix tree, TBB-RR uses a fixed compression budget.

## Definition

For a process node event sequence:

\[
S_v=[x_1,x_2,\ldots,x_n]
\]

given target compression ratio \(C^\star\), define the target keep ratio:

\[
\rho^\star=1-C^\star
\]

and block length:

\[
B=\left\lceil\frac{2}{\rho^\star}\right\rceil.
\]

The sequence is split into non-overlapping blocks:

\[
[a_q,b_q]=[(q-1)B+1,\min(qB,n)].
\]

For each block, keep the first and last events:

\[
P_q=\begin{cases}
\{a_q\}, & b_q=a_q,\\
\{a_q,b_q\}, & b_q>a_q.
\end{cases}
\]

The reduced sequence is:

\[
S'_v=[x_i\mid i\in \cup_q P_q]
\]

in the original order.

## Default

```text
TBB_RR_TARGET_COMPRESSION=0.90
```

This gives:

\[
B=\left\lceil\frac{2}{0.10}\right\rceil=20
\]

and a long-sequence compression ratio near:

\[
1-\frac{2}{20}=0.90.
\]

## Experiment matrix

CADETS:

```bash
DEVICE=0 bash scripts/run_cadets_tbb_rr_verdict.sh
```

THEIA:

```bash
DEVICE=0 bash scripts/run_theia_tbb_rr_verdict.sh
```

Both scripts compare only:

```text
off
prefix_tree
target_boundary
```

with fixed `E1_eha_only`.

## Claim boundary

TBB-RR should be described as a compression-controlled simple reducer.  It is not a confirmed improvement unless paired seeds show stable gains over `prefix_tree` and compression remains close to the prefix-tree level.
