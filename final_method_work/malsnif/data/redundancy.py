from __future__ import annotations

from collections import Counter, OrderedDict
import hashlib
from dataclasses import dataclass, field
import math
from typing import Hashable, Sequence, TypeVar

T = TypeVar("T", bound=Hashable)


@dataclass
class TrieNode:
    children: "OrderedDict[T, TrieNode]" = field(default_factory=OrderedDict)


@dataclass
class WeightedTrieNode:
    """Prefix-tree node that stores how many loop bodies share this prefix."""

    count: int = 0
    children: "OrderedDict[T, WeightedTrieNode]" = field(default_factory=OrderedDict)


def scan_r2l(seq: Sequence[T]) -> int | None:
    seen: set[T] = set()
    lbs: int | None = None
    for i in range(len(seq) - 1, -1, -1):
        if seq[i] in seen:
            lbs = i
        else:
            seen.add(seq[i])
    return lbs


def scan_l2r(seq: Sequence[T]) -> int | None:
    seen: set[T] = set()
    lbe: int | None = None
    for i, x in enumerate(seq):
        if x in seen:
            lbe = i
        else:
            seen.add(x)
    return lbe


def _loop_body_separators(seq: Sequence[T], lbs: int, lbe: int) -> list[int]:
    if lbs < 0 or lbs >= len(seq):
        return []
    anchor = seq[lbs]
    starts = [i for i in range(lbs, lbe + 1) if seq[i] == anchor]
    if not starts or starts[0] != lbs:
        starts.insert(0, lbs)
    return starts


def _build_prefix_tree(seq: Sequence[T], starts: list[int], end_exclusive: int) -> TrieNode:
    root = TrieNode()
    points = starts + [end_exclusive]
    for a, b in zip(points[:-1], points[1:]):
        node = root
        for item in seq[a:b]:
            if item not in node.children:
                node.children[item] = TrieNode()
            node = node.children[item]
    return root


def _preorder(root: TrieNode) -> list[T]:
    """Return preorder traversal of a prefix tree without recursion.

    Real DARPA CDM windows can contain very long repeated process-event
    sequences.  In those cases the prefix tree may degenerate into a chain with
    thousands of nodes.  A recursive DFS then hits Python's recursion limit, so
    this traversal uses an explicit stack while preserving the same insertion
    order as the recursive implementation.
    """
    out: list[T] = []
    stack: list[tuple[T, TrieNode]] = list(reversed(list(root.children.items())))
    while stack:
        item, child = stack.pop()
        out.append(item)
        # OrderedDict preserves insertion order.  Push in reverse so popping
        # visits children in the original recursive preorder.
        if child.children:
            stack.extend(reversed(list(child.children.items())))
    return out


def _build_weighted_prefix_tree(seq: Sequence[T], starts: list[int], end_exclusive: int) -> WeightedTrieNode:
    """Build the MalSnif prefix tree and count shared loop-body prefixes."""

    root = WeightedTrieNode()
    points = starts + [end_exclusive]
    for a, b in zip(points[:-1], points[1:]):
        node = root
        for item in seq[a:b]:
            if item not in node.children:
                node.children[item] = WeightedTrieNode()
            node = node.children[item]
            node.count += 1
    return root


def _preorder_weighted(root: WeightedTrieNode) -> list[tuple[T, int]]:
    """Return preorder traversal as ``(event, prefix_count)`` pairs."""

    out: list[tuple[T, int]] = []
    stack: list[tuple[T, WeightedTrieNode]] = list(reversed(list(root.children.items())))
    while stack:
        item, child = stack.pop()
        out.append((item, int(child.count)))
        if child.children:
            stack.extend(reversed(list(child.children.items())))
    return out


def _counts_to_rhos(counts: Sequence[int]) -> list[float]:
    """Log-normalize prefix multiplicities to [0, 1]."""

    max_count = max([int(c) for c in counts] or [1])
    if max_count <= 1:
        return [0.0 for _ in counts]
    denom = math.log(float(max_count))
    return [max(0.0, min(1.0, math.log(float(max(int(c), 1))) / denom)) for c in counts]


def _merge_recursive_rhos(
    source_tokens: Sequence[T],
    source_rhos: Sequence[float],
    reduced_tokens: Sequence[T],
    reduced_rhos: Sequence[float],
) -> list[float]:
    """Align recursive output to its input and keep the strongest rho seen.

    The recursive prefix-tree reducer preserves relative order but may merge
    repeated items.  A left-to-right greedy alignment is deterministic and keeps
    this weighting side channel independent from labels or risk rules.
    """

    out: list[float] = []
    cursor = 0
    for item, rec_rho in zip(reduced_tokens, reduced_rhos):
        chosen = float(rec_rho)
        found = False
        while cursor < len(source_tokens):
            if source_tokens[cursor] == item:
                chosen = max(chosen, float(source_rhos[cursor]))
                cursor += 1
                found = True
                break
            cursor += 1
        # If alignment cannot find the token (should be rare), keep the recursive
        # rho rather than changing the reduced token sequence.
        out.append(chosen if found else float(rec_rho))
    return out


def redundancy_reduction_weighted_prefix_tree(
    seq: Sequence[T],
    *,
    max_depth: int = 64,
    alpha: float = 0.2,
    return_rho: bool = False,
) -> tuple[list[T], list[float]]:
    """Multiplicity-weighted variant of MalSnif Algorithm 1.

    The event token sequence is identical to the original prefix-tree reduction
    whenever compression happens.  The only extra output is a side-channel weight
    per retained event:

        rho(u) = log(c(u)) / log(max_count)
        omega(u) = 1 + alpha * rho(u)

    where c(u) is how many loop bodies share the trie prefix represented by node
    u.  This preserves repetition intensity without adding new vocabulary items.
    """

    seq = list(seq)
    if len(seq) <= 4 or max_depth <= 0:
        vals = [0.0 if return_rho else 1.0 for _ in seq]
        return seq, vals

    lbs = scan_r2l(seq)
    lbe = scan_l2r(seq)
    if lbs is None or lbe is None or lbs >= lbe:
        vals = [0.0 if return_rho else 1.0 for _ in seq]
        return seq, vals

    init_seq = seq[:lbs]
    end_seq = seq[lbe + 1 :]
    starts = _loop_body_separators(seq, lbs, lbe)
    if len(starts) <= 1:
        vals = [0.0 if return_rho else 1.0 for _ in seq]
        return seq, vals

    tree = _build_weighted_prefix_tree(seq, starts, lbe + 1)
    pairs = _preorder_weighted(tree)
    slb = [item for item, _ in pairs]
    slb_rhos = _counts_to_rhos([cnt for _, cnt in pairs])

    loop_len = lbe + 1 - lbs
    if len(slb) >= loop_len:
        middle = slb
        middle_rhos = slb_rhos
    else:
        rec_tokens, rec_rhos = redundancy_reduction_weighted_prefix_tree(
            slb, max_depth=max_depth - 1, alpha=alpha, return_rho=True
        )
        middle = rec_tokens
        middle_rhos = _merge_recursive_rhos(slb, slb_rhos, rec_tokens, rec_rhos)

    reduced = init_seq + middle + end_seq
    reduced_rhos = [0.0] * len(init_seq) + middle_rhos + [0.0] * len(end_seq)
    if len(reduced) >= len(seq):
        vals = [0.0 if return_rho else 1.0 for _ in seq]
        return seq, vals

    if return_rho:
        return reduced, [max(0.0, min(1.0, float(r))) for r in reduced_rhos]
    a = max(0.0, float(alpha))
    return reduced, [1.0 + a * max(0.0, min(1.0, float(r))) for r in reduced_rhos]



def _blocks_equal(keys: Sequence[Hashable], a: int, b: int, length: int) -> bool:
    """Return True when two adjacent candidate blocks are exactly equal."""
    # length is intentionally small (default <=16), so this explicit loop is
    # faster and avoids allocating temporary slices for every comparison.
    if b + length > len(keys):
        return False
    for off in range(length):
        if keys[a + off] != keys[b + off]:
            return False
    return True


def _repeat_weight(repeat_count: int, *, repeat_cap: int = 32, alpha: float = 0.3) -> float:
    """Convert a tandem-repeat count into a bounded numeric side weight."""
    repeat_count = max(1, int(repeat_count))
    repeat_cap = max(2, int(repeat_cap))
    a = max(0.0, float(alpha))
    if repeat_count <= 1:
        return 1.0
    rho = min(1.0, math.log(float(repeat_count)) / math.log(float(repeat_cap)))
    return 1.0 + a * rho


def redundancy_reduction_bounded_tandem_repeat(
    seq: Sequence[T],
    *,
    max_block_len: int = 16,
    min_gain: int = 2,
    repeat_cap: int = 32,
    alpha: float = 0.3,
) -> tuple[list[T], list[float]]:
    """Bounded tandem-repeat redundancy reduction (BTR-RR).

    This is a conservative replacement for MalSnif's prefix-tree sequence
    reducer.  Instead of segmenting a process trace into loop bodies and merging
    shared prefixes, BTR-RR compresses only exact consecutive repeated blocks:

        A + B^r + C  ->  A + B + C

    The retained copy of B receives a bounded logarithmic repeat weight.  The
    event tokens are not changed, so Word2Vec/MCBG vocabulary remains stable.
    The left-to-right, max-gain rule is deterministic and intentionally avoids
    crossing inserted variant events, which may be attack evidence.
    """

    seq = list(seq)
    n = len(seq)
    if n <= 1:
        return seq, [1.0 for _ in seq]

    Lmax = max(1, int(max_block_len))
    min_gain = max(0, int(min_gain))
    out: list[T] = []
    weights: list[float] = []

    # In the current project event ids are tuples of already sanitized event
    # tokens.  Keeping the exact key avoids overly aggressive semantic merging.
    keys: list[Hashable] = [x for x in seq]

    i = 0
    while i < n:
        best_gain = 0
        best_l = 0
        best_r = 1
        max_l_here = min(Lmax, (n - i) // 2)
        for l in range(1, max_l_here + 1):
            r = 1
            # Count how many consecutive l-sized blocks equal the first block.
            while i + (r + 1) * l <= n and _blocks_equal(keys, i, i + r * l, l):
                r += 1
            if r <= 1:
                continue
            gain = (r - 1) * l
            # Prefer the strongest compression; on ties prefer the shorter
            # period (e.g., AAAAAA -> A^6 rather than (AA)^3).
            if gain > best_gain or (gain == best_gain and best_l and l < best_l):
                best_gain = gain
                best_l = l
                best_r = r

        if best_l > 0 and best_gain >= min_gain:
            w = _repeat_weight(best_r, repeat_cap=repeat_cap, alpha=alpha)
            for item in seq[i : i + best_l]:
                out.append(item)
                weights.append(w)
            i += best_l * best_r
        else:
            out.append(seq[i])
            weights.append(1.0)
            i += 1

    if len(out) >= n:
        return seq, [1.0 for _ in seq]
    return out, weights



def _skeleton_weight(phrase_len: int, *, max_phrase_len: int = 24, alpha: float = 0.25) -> float:
    """Convert LZ-SR skeletonization strength into a bounded side weight."""
    phrase_len = max(1, int(phrase_len))
    max_phrase_len = max(3, int(max_phrase_len))
    dropped = max(0, phrase_len - 2)
    max_dropped = max(1, max_phrase_len - 2)
    if dropped <= 0:
        return 1.0
    rho = min(1.0, math.log1p(float(dropped)) / math.log1p(float(max_dropped)))
    return 1.0 + max(0.0, float(alpha)) * rho


def _insert_lz_history(
    history: dict[tuple[Hashable, ...], list[int]],
    keys: Sequence[Hashable],
    start: int,
    end_exclusive: int,
    *,
    max_phrase_len: int,
) -> None:
    """Index phrases whose start lies in [start, end_exclusive)."""
    n = len(keys)
    for pos in range(max(0, start), min(n, end_exclusive)):
        max_l = min(int(max_phrase_len), n - pos)
        for length in range(1, max_l + 1):
            phrase = tuple(keys[pos : pos + length])
            history.setdefault(phrase, []).append(pos)


def _has_recent_lz_match(
    history: dict[tuple[Hashable, ...], list[int]],
    phrase: tuple[Hashable, ...],
    *,
    current_pos: int,
    window: int,
) -> bool:
    """Return True if phrase has a previous start within the look-back window."""
    positions = history.get(phrase)
    if not positions:
        return False
    lower = max(0, int(current_pos) - max(1, int(window)))
    # Positions are appended in increasing order.  Walk backward only while a
    # recent match is plausible; lists are short for most phrases in practice.
    for pos in reversed(positions):
        if pos >= current_pos:
            continue
        if pos < lower:
            return False
        return True
    return False


def redundancy_reduction_lz_skeleton(
    seq: Sequence[T],
    *,
    min_phrase_len: int = 4,
    max_phrase_len: int = 24,
    window: int = 512,
    min_gain: int = 2,
    alpha: float = 0.25,
) -> tuple[list[T], list[float]]:
    """Bounded Lempel-Ziv Skeleton Reduction (LZ-SR).

    LZ-SR is a lightweight replacement candidate for MalSnif Algorithm 1.  It
    scans a process event sequence from left to right and compresses only exact
    phrases that have already appeared in the same process sequence.  The first
    occurrence of a phrase is preserved in full; later exact matches keep only
    their boundary events.  A numeric side weight records the compression
    strength, so no new event tokens or vocabulary entries are introduced.

        A + P + ... + P + C  ->  A + P + ... + boundary(P) + C

    The exact-match condition is intentionally conservative: variants with an
    inserted or substituted event are not skeletonized, which helps preserve
    potentially malicious deviations from repeated benign behavior.
    """
    seq = list(seq)
    n = len(seq)
    if n <= 1:
        return seq, [1.0 for _ in seq]

    Lmin = max(1, int(min_phrase_len))
    Lmax = max(Lmin, int(max_phrase_len))
    lookback = max(1, int(window))
    min_gain = max(0, int(min_gain))

    keys: list[Hashable] = [x for x in seq]
    history: dict[tuple[Hashable, ...], list[int]] = {}
    out: list[T] = []
    weights: list[float] = []

    i = 0
    while i < n:
        best_l = 0
        upper_l = min(Lmax, n - i)
        for length in range(upper_l, Lmin - 1, -1):
            if length - 2 < min_gain:
                continue
            phrase = tuple(keys[i : i + length])
            if _has_recent_lz_match(history, phrase, current_pos=i, window=lookback):
                best_l = length
                break

        if best_l > 0:
            w = _skeleton_weight(best_l, max_phrase_len=Lmax, alpha=alpha)
            out.append(seq[i])
            weights.append(w)
            # best_l >= Lmin >= 2 in the intended configuration, but keep the
            # guard to avoid duplicating a one-token phrase if parameters change.
            if best_l > 1:
                out.append(seq[i + best_l - 1])
                weights.append(w)
            _insert_lz_history(history, keys, i, i + best_l, max_phrase_len=Lmax)
            i += best_l
        else:
            out.append(seq[i])
            weights.append(1.0)
            _insert_lz_history(history, keys, i, i + 1, max_phrase_len=Lmax)
            i += 1

    if len(out) >= n:
        return seq, [1.0 for _ in seq]
    return out, weights




def _stable_u64(value: object) -> int:
    """Return a deterministic 64-bit hash value.

    Python's built-in hash is intentionally randomized between processes, so
    WRA-RR uses blake2b from the standard library to keep preprocessing
    reproducible across runs and platforms.
    """
    data = repr(value).encode("utf-8", errors="surrogatepass")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def _wra_event_key(item: Hashable) -> Hashable:
    """Event key for WRA-RR anchors.

    We keep the current sanitized event token tuple instead of labels, risk
    scores, or external rules.  The position is added later to avoid pathological
    hash ties on long identical runs while still keeping selection deterministic.
    """
    return tuple(_tokens_of(item))


def redundancy_reduction_winnowing_anchor(
    seq: Sequence[T],
    *,
    window: int = 11,
) -> tuple[list[T], list[float]]:
    """Winnowing Representative Anchor Redundancy Reduction (WRA-RR).

    WRA-RR is a one-parameter, local replacement candidate for MalSnif
    Algorithm 1.  It scans a process-event sequence using a fixed-size sliding
    window and selects the event with the minimum deterministic event-position
    hash in each window.  Selected anchors are deduplicated, sorted by original
    order, and the first/last events are always retained.

    No new event tokens or numeric weights are introduced; the window size
    controls the compression ratio.  With random-like hashes, the expected
    anchor density is approximately 2/(window+1).
    """
    seq = list(seq)
    n = len(seq)
    w = max(2, int(window))
    if n <= w:
        return seq, [1.0 for _ in seq]

    # Event-position hashing avoids very high anchor density when a run contains
    # many identical event keys.  This keeps the reduction ratio controlled by w.
    hashes = [_stable_u64((_wra_event_key(item), i)) for i, item in enumerate(seq)]
    anchors: set[int] = {0, n - 1}

    for start in range(0, n - w + 1):
        end = start + w
        min_hash = min(hashes[start:end])
        # Rightmost minimum is deterministic and follows the winnowing convention.
        for pos in range(end - 1, start - 1, -1):
            if hashes[pos] == min_hash:
                anchors.add(pos)
                break

    out = [seq[i] for i in sorted(anchors)]
    if len(out) >= n:
        return seq, [1.0 for _ in seq]
    return out, [1.0 for _ in out]


def redundancy_reduction_target_boundary(
    seq: Sequence[T],
    *,
    target_compression: float = 0.90,
) -> tuple[list[T], list[float]]:
    """Target-Budget Boundary Redundancy Reduction (TBB-RR).

    TBB-RR is an intentionally simple replacement candidate for MalSnif
    Algorithm 1.  Given a target compression ratio C*, the event sequence is
    split into fixed-size, non-overlapping blocks.  Each block contributes its
    first and last events, preserving temporal order and avoiding new event
    tokens, risk rules, event-family heuristics, historical dictionaries, or
    prefix trees.

        block length B = ceil(2 / (1 - C*))

    Since each full block keeps two events, the realized compression ratio is
    approximately C* for long sequences.  Singleton tail blocks keep their only
    event.
    """
    seq = list(seq)
    n = len(seq)
    if n <= 2:
        return seq, [1.0 for _ in seq]

    try:
        c_star = float(target_compression)
    except Exception:
        c_star = 0.90
    # Keep the method in the intended high-compression but non-degenerate range.
    c_star = max(0.0, min(0.98, c_star))
    keep_rate = max(1.0e-6, 1.0 - c_star)
    # Subtract a tiny epsilon before ceil so common decimal budgets such as
    # C*=0.90 produce B=20 rather than B=21 due to binary floating error.
    block_len = max(2, int(math.ceil((2.0 / keep_rate) - 1.0e-9)))
    if n <= block_len:
        return seq, [1.0 for _ in seq]

    anchors: list[int] = []
    for start in range(0, n, block_len):
        end = min(start + block_len - 1, n - 1)
        anchors.append(start)
        if end != start:
            anchors.append(end)

    out = [seq[i] for i in anchors]
    if len(out) >= n:
        return seq, [1.0 for _ in seq]
    return out, [1.0 for _ in out]

def _frc_resource_family(tokens: Sequence[str]) -> str:
    """Coarse destination/resource family for FRC-RR.

    Event tokens are [edge_type, suffix, path_tokens...].  We intentionally use
    only operation/resource-family information, not labels or risk rules.
    """
    if not tokens:
        return "event"
    event = str(tokens[0]).lower()
    suffix = str(tokens[1]).lower() if len(tokens) > 1 else ""
    joined = "/".join(str(t).lower() for t in tokens)
    if "reg" in event or "registry" in joined or joined.startswith("hk"):
        return "registry"
    if any(term in event for term in _NETWORK_TERMS) or "<ip>" in joined or "socket" in joined:
        return "network"
    if any(term in event for term in _PROCESS_TERMS) or suffix in _EXEC_SUFFIXES:
        return "process"
    if suffix and suffix != "<nosuffix>":
        return "file"
    return "object"


def _family_run_key(item: Hashable) -> Hashable:
    """Return the operation-resource family used by FRC-RR.

    This approximates phi(x)=<EdgeType,DstType> with the current sanitized event
    token representation: the first token is EdgeType and the second/path tokens
    allow a coarse destination-resource family to be inferred.  Exact paths and
    object ids are deliberately ignored so write/read/query bursts can be capped
    even when each concrete object differs.
    """
    toks = _tokens_of(item)
    edge = toks[0] if toks else "<event>"
    return (edge, _frc_resource_family(toks))


def _run_weight(run_len: int, *, repeat_cap: int = 32, alpha: float = 0.25) -> float:
    """Log-normalized run-length side weight for FRC-RR."""
    run_len = max(1, int(run_len))
    repeat_cap = max(2, int(repeat_cap))
    if run_len <= 1:
        return 1.0
    rho = min(1.0, math.log(float(run_len)) / math.log(float(repeat_cap)))
    return 1.0 + max(0.0, float(alpha)) * rho


def _run_representative_positions(start: int, end: int, cap_size: int) -> list[int]:
    """Return representative positions for a long run, preserving order.

    Default cap_size=3 gives first/middle/last, matching the paper-friendly
    FRC-RR definition.  The general form keeps evenly spaced representatives so
    changing the cap remains deterministic without adding another mechanism.
    """
    cap = max(1, int(cap_size))
    length = end - start + 1
    if length <= cap:
        return list(range(start, end + 1))
    if cap == 1:
        return [start]
    if cap == 2:
        return [start, end]
    positions = [round(start + i * (length - 1) / (cap - 1)) for i in range(cap)]
    out: list[int] = []
    seen: set[int] = set()
    for pos in positions:
        pos = max(start, min(end, int(pos)))
        if pos not in seen:
            out.append(pos)
            seen.add(pos)
    return out


def redundancy_reduction_family_run_cap(
    seq: Sequence[T],
    *,
    cap_size: int = 3,
    repeat_cap: int = 32,
    alpha: float = 0.25,
) -> tuple[list[T], list[float]]:
    """Family-Run Capping Redundancy Reduction (FRC-RR).

    FRC-RR is a deliberately small replacement candidate for MalSnif Algorithm 1.
    Consecutive events with the same operation-resource family are grouped into
    a run.  Short runs are kept unchanged.  Long runs are capped by retaining a
    few representatives (default first/middle/last) and attaching a bounded
    logarithmic run-length weight to those representatives.

        x_a ... x_b with phi(x_t)=g and r=b-a+1>K
        -> x_a, x_mid, x_b with omega=1+alpha*log(r)/log(Rcap)

    No new event tokens are introduced, event order is preserved, and the rule is
    a single linear scan over each process event sequence.
    """
    seq = list(seq)
    n = len(seq)
    if n <= 1:
        return seq, [1.0 for _ in seq]
    cap = max(1, int(cap_size))
    out: list[T] = []
    weights: list[float] = []
    i = 0
    while i < n:
        key = _family_run_key(seq[i])
        start = i
        i += 1
        while i < n and _family_run_key(seq[i]) == key:
            i += 1
        end = i - 1
        run_len = end - start + 1
        if run_len <= cap:
            for item in seq[start : end + 1]:
                out.append(item)
                weights.append(1.0)
        else:
            w = _run_weight(run_len, repeat_cap=repeat_cap, alpha=alpha)
            for pos in _run_representative_positions(start, end, cap):
                out.append(seq[pos])
                weights.append(w)
    if len(out) >= n:
        return seq, [1.0 for _ in seq]
    return out, weights


def redundancy_reduction_first_last_boundary(
    seq: Sequence[T],
    *,
    repeat_cap: int = 32,
    alpha: float = 0.25,
) -> tuple[list[T], list[float]]:
    """First-Last Boundary Redundancy Reduction (FLB-RR).

    FLB-RR is a minimal run-based replacement candidate for MalSnif Algorithm 1.
    Consecutive events with the same operation-resource family are grouped into
    a run.  A singleton run is kept unchanged; a longer run is represented only
    by its first and last boundary events.  The retained boundary events receive
    a bounded logarithmic run-length weight.

        x_a ... x_b with phi(x_t)=g and r=b-a+1>1
        -> x_a, x_b with omega=1+alpha*log(r)/log(Rcap)

    The rule is a single linear scan, preserves event order, and introduces no
    new event tokens or vocabulary entries.
    """

    seq = list(seq)
    n = len(seq)
    if n <= 1:
        return seq, [1.0 for _ in seq]

    out: list[T] = []
    weights: list[float] = []
    i = 0
    while i < n:
        key = _family_run_key(seq[i])
        start = i
        i += 1
        while i < n and _family_run_key(seq[i]) == key:
            i += 1
        end = i - 1
        run_len = end - start + 1
        if run_len == 1:
            out.append(seq[start])
            weights.append(1.0)
        else:
            w = _run_weight(run_len, repeat_cap=repeat_cap, alpha=alpha)
            out.append(seq[start])
            weights.append(w)
            out.append(seq[end])
            weights.append(w)

    return out, weights

def redundancy_reduction_prefix_tree(seq: Sequence[T], max_depth: int = 64) -> list[T]:
    """Prefix-tree based process event sequence redundancy reduction.

    This follows the MalSnif paper's idea of compressing repeated process event
    loops via prefix-tree traversal.  The implementation is deliberately safe for
    long DARPA CDM sequences: tree traversal is iterative, and recursive
    re-compression is capped by ``max_depth``.

    Paper example:
        ABCDEDECDEKEHI -> ABCDEKEHI
    """
    seq = list(seq)
    if len(seq) <= 4 or max_depth <= 0:
        return seq
    lbs = scan_r2l(seq)
    lbe = scan_l2r(seq)
    if lbs is None or lbe is None or lbs >= lbe:
        return seq
    init_seq = seq[:lbs]
    end_seq = seq[lbe + 1 :]
    starts = _loop_body_separators(seq, lbs, lbe)
    if len(starts) <= 1:
        return seq
    tree = _build_prefix_tree(seq, starts, lbe + 1)
    slb = _preorder(tree)
    if len(slb) >= (lbe + 1 - lbs):
        # Avoid infinite recursion on sequences whose prefix-tree traversal is
        # not shorter than the loop body.
        middle = slb
    else:
        middle = redundancy_reduction_prefix_tree(slb, max_depth=max_depth - 1)
    reduced = init_seq + middle + end_seq
    if len(reduced) >= len(seq):
        return seq
    return reduced


_PROCESS_TERMS = (
    "exec",
    "execute",
    "fork",
    "clone",
    "unit",
    "loadlibrary",
    "load_library",
    "mmap",
)
_NETWORK_TERMS = (
    "connect",
    "socket",
    "send",
    "sendto",
    "sendmsg",
    "recv",
    "recvfrom",
    "recvmsg",
    "accept",
    "bind",
    "listen",
)
_REGISTRY_TERMS = ("registry", "reg_", "setvalue", "set_value", "setkey")
_WRITE_TERMS = ("write", "rename", "modify", "chmod", "unlink", "delete", "link")
_EXEC_SUFFIXES = {
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bat",
    ".cmd",
    ".ps1",
    ".vbs",
    ".js",
    ".jar",
    ".sh",
    ".py",
    ".pl",
    ".rb",
    ".elf",
    ".scr",
    ".sys",
}
_SENSITIVE_PATH_TERMS = (
    "tmp",
    "<tmp>",
    "[temp]",
    "startup",
    "runonce",
    "autorun",
    "system32",
    "syswow64",
    "program_files",
    "bin",
    "cron",
    "init.d",
    "service",
    "powershell",
    "cmd",
    "bash",
)


def _tokens_of(item: Hashable) -> list[str]:
    if isinstance(item, (tuple, list)):
        return [str(x).lower() for x in item]
    return [str(item).lower()]


def event_risk_score(item: Hashable) -> float:
    """Heuristic risk score from MalSnif event tokens.

    The graph builder represents an event as
    ``[event_type, suffix, path_token_1, ...]``.  This score intentionally uses
    only that local token information so the optimized reduction remains a
    preprocessing-only change and does not depend on labels.
    """
    toks = _tokens_of(item)
    if not toks:
        return 0.0
    event_type = toks[0]
    suffix = toks[1] if len(toks) > 1 else ""
    joined = "/".join(toks)
    score = 0.0
    if any(term in event_type for term in _PROCESS_TERMS):
        score += 3.0
    if any(term in event_type for term in _NETWORK_TERMS):
        score += 2.5
    if any(term in event_type for term in _REGISTRY_TERMS) or "registry" in joined:
        score += 2.5
    if any(term in event_type for term in _WRITE_TERMS):
        score += 1.0
    if suffix in _EXEC_SUFFIXES:
        score += 1.75
    if any(term in joined for term in _SENSITIVE_PATH_TERMS):
        score += 1.0
    return score


def _event_family(item: Hashable) -> str:
    toks = _tokens_of(item)
    if not toks:
        return "<event>"
    return toks[0].replace(" ", "_")[:48]


def _log2_bucket(n: int) -> int:
    return max(0, min(16, int(math.ceil(math.log2(max(int(n), 1))))))


def _repeat_summary_token(item: T, original_count: int, kept_count: int) -> T:
    family = _event_family(item)
    dropped = max(0, int(original_count) - int(kept_count))
    payload = (
        "<rep>",
        family,
        f"<cnt_log2_{_log2_bucket(original_count)}>",
        f"<drop_log2_{_log2_bucket(dropped)}>",
    )
    if isinstance(item, tuple):
        return payload  # type: ignore[return-value]
    return f"<rep:{family}:cnt_log2_{_log2_bucket(original_count)}:drop_log2_{_log2_bucket(dropped)}>"  # type: ignore[return-value]


def _is_summary_token(item: Hashable) -> bool:
    if isinstance(item, tuple) and item:
        return str(item[0]) == "<rep>"
    return str(item).startswith("<rep:")


def _trim_to_budget(seq: list[T], max_events: int | None, risk_threshold: float) -> list[T]:
    if max_events is None or int(max_events) <= 0 or len(seq) <= int(max_events):
        return seq
    budget = int(max_events)
    if budget <= 2:
        return seq[:budget]
    priority = {
        0,
        len(seq) - 1,
        *(
            i
            for i, item in enumerate(seq)
            if _is_summary_token(item) or event_risk_score(item) >= risk_threshold
        ),
    }
    if len(priority) > budget:
        stride = len(priority) / budget
        ordered = sorted(priority)
        chosen = {ordered[min(len(ordered) - 1, int(i * stride))] for i in range(budget)}
        return [seq[i] for i in sorted(chosen)]

    chosen = set(priority)
    remaining = budget - len(chosen)
    candidates = [i for i in range(len(seq)) if i not in chosen]
    if remaining > 0 and candidates:
        stride = len(candidates) / remaining
        for i in range(remaining):
            chosen.add(candidates[min(len(candidates) - 1, int(i * stride))])
    return [seq[i] for i in sorted(chosen)]


def redundancy_reduction_risk_time_prefix_tree(
    seq: Sequence[T],
    *,
    max_depth: int = 64,
    max_events: int | None = None,
    risk_threshold: float = 2.5,
    preserve_risk_events: int = 1,
    repeat_summary: bool = True,
    repeat_min: int = 3,
) -> list[T]:
    """Risk/time preserving variant of the MalSnif prefix-tree reducer.

    It first applies the original prefix-tree compression, then reconstructs a
    chronological subsequence that preserves:
    - the compressed event multiset,
    - a small quota of extra high-risk repeated events,
    - optional repeat summary tokens for events whose counts were compressed.

    This keeps the change small: model inputs are still ordinary event-token
    sequences, and the original Algorithm 1 remains available unchanged.
    """
    original = list(seq)
    if not original:
        return original
    base = redundancy_reduction_prefix_tree(original, max_depth=max_depth)
    if len(base) >= len(original):
        return _trim_to_budget(base, max_events, risk_threshold)

    base_count = Counter(base)
    base_remaining = Counter(base)
    base_positions: set[int] = set()
    dropped_positions_by_item: dict[T, list[int]] = {}
    for pos, item in enumerate(original):
        if base_remaining[item] > 0:
            base_positions.add(pos)
            base_remaining[item] -= 1
        else:
            dropped_positions_by_item.setdefault(item, []).append(pos)

    protected_positions: set[int] = set()
    quota = max(0, int(preserve_risk_events))
    if quota > 0:
        for item, positions in dropped_positions_by_item.items():
            if event_risk_score(item) >= risk_threshold:
                protected_positions.update(positions[-quota:])

    last_pos: dict[T, int] = {}
    for pos, item in enumerate(original):
        last_pos[item] = pos

    summaries: dict[int, list[T]] = {}
    if repeat_summary:
        original_count = Counter(original)
        for item, cnt in original_count.items():
            kept = int(base_count.get(item, 0)) + sum(1 for p in protected_positions if original[p] == item)
            dropped = int(cnt) - kept
            if cnt >= int(repeat_min) and dropped > 0:
                summaries.setdefault(last_pos[item], []).append(_repeat_summary_token(item, cnt, kept))

    out: list[T] = []
    for pos, item in enumerate(original):
        if pos in base_positions or pos in protected_positions:
            out.append(item)
        if pos in summaries:
            out.extend(summaries[pos])
    return _trim_to_budget(out, max_events, risk_threshold)


def reduce_event_sequence(
    seq: Sequence[T],
    *,
    mode: str = "prefix_tree",
    max_depth: int = 64,
    max_events: int | None = None,
    risk_threshold: float = 2.5,
    preserve_risk_events: int = 1,
    repeat_summary: bool = True,
    repeat_min: int = 3,
    mw_prr_alpha: float = 0.2,
    btr_rr_max_block_len: int = 16,
    btr_rr_min_gain: int = 2,
    btr_rr_repeat_cap: int = 32,
    btr_rr_alpha: float = 0.3,
    lz_sr_min_phrase_len: int = 4,
    lz_sr_max_phrase_len: int = 24,
    lz_sr_window: int = 512,
    lz_sr_min_gain: int = 2,
    lz_sr_alpha: float = 0.25,
    frc_rr_cap_size: int = 3,
    frc_rr_repeat_cap: int = 32,
    frc_rr_alpha: float = 0.25,
    flb_rr_repeat_cap: int = 32,
    flb_rr_alpha: float = 0.25,
    wra_rr_window: int = 11,
    tbb_rr_target_compression: float = 0.90,
) -> list[T]:
    mode = (mode or "prefix_tree").lower()
    if mode in {"off", "none", "false", "no"}:
        return list(seq)
    if mode in {"prefix_tree", "paper", "algorithm1", "algorithm_1"}:
        return redundancy_reduction_prefix_tree(seq, max_depth=max_depth)
    if mode in {"weighted_prefix_tree", "mw_prefix_tree", "mw_prr", "multiplicity_weighted_prefix_tree"}:
        reduced, _weights = redundancy_reduction_weighted_prefix_tree(seq, max_depth=max_depth, alpha=mw_prr_alpha)
        return reduced
    if mode in {"btr_rr", "bounded_tandem_repeat", "tandem_repeat", "bounded_tandem_repeat_rr"}:
        reduced, _weights = redundancy_reduction_bounded_tandem_repeat(
            seq,
            max_block_len=btr_rr_max_block_len,
            min_gain=btr_rr_min_gain,
            repeat_cap=btr_rr_repeat_cap,
            alpha=btr_rr_alpha,
        )
        return reduced
    if mode in {"lz_skeleton", "lz_sr", "bounded_lz_skeleton", "history_phrase_skeleton"}:
        reduced, _weights = redundancy_reduction_lz_skeleton(
            seq,
            min_phrase_len=lz_sr_min_phrase_len,
            max_phrase_len=lz_sr_max_phrase_len,
            window=lz_sr_window,
            min_gain=lz_sr_min_gain,
            alpha=lz_sr_alpha,
        )
        return reduced
    if mode in {"family_run_cap", "frc_rr", "frc", "run_cap", "family_run_capping"}:
        reduced, _weights = redundancy_reduction_family_run_cap(
            seq,
            cap_size=frc_rr_cap_size,
            repeat_cap=frc_rr_repeat_cap,
            alpha=frc_rr_alpha,
        )
        return reduced
    if mode in {"first_last_boundary", "flb_rr", "flb", "first_last", "boundary_run"}:
        reduced, _weights = redundancy_reduction_first_last_boundary(
            seq,
            repeat_cap=flb_rr_repeat_cap,
            alpha=flb_rr_alpha,
        )
        return reduced
    if mode in {"winnowing_anchor", "wra_rr", "wra", "representative_anchor", "winnowing"}:
        reduced, _weights = redundancy_reduction_winnowing_anchor(
            seq,
            window=wra_rr_window,
        )
        return reduced
    if mode in {"target_boundary", "tbb_rr", "tbb", "target_budget", "target_budget_boundary"}:
        reduced, _weights = redundancy_reduction_target_boundary(
            seq,
            target_compression=tbb_rr_target_compression,
        )
        return reduced
    if mode in {"risk_time_prefix_tree", "risk_time", "rt_prefix_tree", "rtprr"}:
        return redundancy_reduction_risk_time_prefix_tree(
            seq,
            max_depth=max_depth,
            max_events=max_events,
            risk_threshold=risk_threshold,
            preserve_risk_events=preserve_risk_events,
            repeat_summary=repeat_summary,
            repeat_min=repeat_min,
        )
    raise ValueError(f"Unknown redundancy_mode={mode!r}")


def reduce_event_sequence_with_weights(
    seq: Sequence[T],
    *,
    mode: str = "prefix_tree",
    max_depth: int = 64,
    max_events: int | None = None,
    risk_threshold: float = 2.5,
    preserve_risk_events: int = 1,
    repeat_summary: bool = True,
    repeat_min: int = 3,
    mw_prr_alpha: float = 0.2,
    btr_rr_max_block_len: int = 16,
    btr_rr_min_gain: int = 2,
    btr_rr_repeat_cap: int = 32,
    btr_rr_alpha: float = 0.3,
    lz_sr_min_phrase_len: int = 4,
    lz_sr_max_phrase_len: int = 24,
    lz_sr_window: int = 512,
    lz_sr_min_gain: int = 2,
    lz_sr_alpha: float = 0.25,
    frc_rr_cap_size: int = 3,
    frc_rr_repeat_cap: int = 32,
    frc_rr_alpha: float = 0.25,
    flb_rr_repeat_cap: int = 32,
    flb_rr_alpha: float = 0.25,
    wra_rr_window: int = 11,
    tbb_rr_target_compression: float = 0.90,
) -> tuple[list[T], list[float]]:
    """Reduce a process-event sequence and return per-event MW-PRR weights.

    Non-weighted modes return all-one weights, preserving backward behavior.
    ``weighted_prefix_tree`` returns the same token sequence as prefix-tree
    compression but attaches multiplicity weights in a separate numeric channel.
    """

    mode_l = (mode or "prefix_tree").lower()
    if mode_l in {"weighted_prefix_tree", "mw_prefix_tree", "mw_prr", "multiplicity_weighted_prefix_tree"}:
        return redundancy_reduction_weighted_prefix_tree(seq, max_depth=max_depth, alpha=mw_prr_alpha)
    if mode_l in {"btr_rr", "bounded_tandem_repeat", "tandem_repeat", "bounded_tandem_repeat_rr"}:
        return redundancy_reduction_bounded_tandem_repeat(
            seq,
            max_block_len=btr_rr_max_block_len,
            min_gain=btr_rr_min_gain,
            repeat_cap=btr_rr_repeat_cap,
            alpha=btr_rr_alpha,
        )
    if mode_l in {"lz_skeleton", "lz_sr", "bounded_lz_skeleton", "history_phrase_skeleton"}:
        return redundancy_reduction_lz_skeleton(
            seq,
            min_phrase_len=lz_sr_min_phrase_len,
            max_phrase_len=lz_sr_max_phrase_len,
            window=lz_sr_window,
            min_gain=lz_sr_min_gain,
            alpha=lz_sr_alpha,
        )
    if mode_l in {"family_run_cap", "frc_rr", "frc", "run_cap", "family_run_capping"}:
        return redundancy_reduction_family_run_cap(
            seq,
            cap_size=frc_rr_cap_size,
            repeat_cap=frc_rr_repeat_cap,
            alpha=frc_rr_alpha,
        )
    if mode_l in {"first_last_boundary", "flb_rr", "flb", "first_last", "boundary_run"}:
        return redundancy_reduction_first_last_boundary(
            seq,
            repeat_cap=flb_rr_repeat_cap,
            alpha=flb_rr_alpha,
        )
    if mode_l in {"winnowing_anchor", "wra_rr", "wra", "representative_anchor", "winnowing"}:
        return redundancy_reduction_winnowing_anchor(
            seq,
            window=wra_rr_window,
        )
    if mode_l in {"target_boundary", "tbb_rr", "tbb", "target_budget", "target_budget_boundary"}:
        return redundancy_reduction_target_boundary(
            seq,
            target_compression=tbb_rr_target_compression,
        )
    reduced = reduce_event_sequence(
        seq,
        mode=mode_l,
        max_depth=max_depth,
        max_events=max_events,
        risk_threshold=risk_threshold,
        preserve_risk_events=preserve_risk_events,
        repeat_summary=repeat_summary,
        repeat_min=repeat_min,
        mw_prr_alpha=mw_prr_alpha,
        btr_rr_max_block_len=btr_rr_max_block_len,
        btr_rr_min_gain=btr_rr_min_gain,
        btr_rr_repeat_cap=btr_rr_repeat_cap,
        btr_rr_alpha=btr_rr_alpha,
        lz_sr_min_phrase_len=lz_sr_min_phrase_len,
        lz_sr_max_phrase_len=lz_sr_max_phrase_len,
        lz_sr_window=lz_sr_window,
        lz_sr_min_gain=lz_sr_min_gain,
        lz_sr_alpha=lz_sr_alpha,
        frc_rr_cap_size=frc_rr_cap_size,
        frc_rr_repeat_cap=frc_rr_repeat_cap,
        frc_rr_alpha=frc_rr_alpha,
        flb_rr_repeat_cap=flb_rr_repeat_cap,
        flb_rr_alpha=flb_rr_alpha,
        wra_rr_window=wra_rr_window,
        tbb_rr_target_compression=tbb_rr_target_compression,
    )
    return reduced, [1.0 for _ in reduced]
