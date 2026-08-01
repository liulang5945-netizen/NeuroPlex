"""Side-channel topology builder — S7 修复：结构性拓扑替代全连接 mesh。

Replaces full-mesh N×(N-1) channel establishment with structured topologies
driven by NeuronGeometry distance and spec capacity.

Modes:
  full       : N×(N-1) full mesh (backward compat)
  knn        : k-nearest neighbors by geometry distance (symmetric)
  hub_spoke  : largest-spec neuron(s) as hub(s); non-hubs connect only via hub
  hybrid     : same-(domain,spec) full mesh + cross-spec via spec-hub +
               cross-domain via global-hub  ← default, highest upper limit

The hybrid mode mirrors cortical hierarchy:
  1. Same (domain, spec)  → full mesh (local microcircuit, tightest coupling)
  2. Same domain, diff spec → via spec-hub (largest spec in domain = association cortex)
  3. Cross-domain          → via global-hub (largest spec globally = prefrontal analog)

All connections get distance-gated init_scale: near neighbors → stronger prior.
This gives each channel a clearer role instead of gradient being evenly split
across N×(N-1) undifferentiated connections.

Usage:
    from taiji.resonance.topology import (
        build_topology, establish_topology_channels,
        infer_topology_from_state, topology_summary,
    )

    # Training: build + establish
    topology = build_topology(neurons, geometry, mode="hybrid")
    stats = establish_topology_channels(neurons, topology, geometry)

    # Eval: auto-infer from checkpoint to match training topology
    topology = infer_topology_from_state(ckpt["side_channels_state"])
    establish_topology_channels(neurons, topology, geometry)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import torch

from .geometry import NeuronGeometry


# ── Hub selection helpers ────────────────────────────────────────────────

def _spec_capacity(neuron) -> int:
    """Approximate capacity metric for hub selection (higher = more hub-worthy).

    Uses hidden_size × num_hidden_layers as a simple capacity proxy.
    Standard (768×10=7680) > Compact (512×6=3072) > Foundation (384×6=2304).
    """
    c = neuron.config
    return c.hidden_size * c.num_hidden_layers


def _domain_of(nid: str) -> str:
    """Extract domain from neuron ID: 'zh_aug0_dialogue' → 'zh'."""
    return nid.split("_")[0] if "_" in nid else nid


# ── Topology builders ────────────────────────────────────────────────────

def build_topology(
    neurons: Dict[str, object],
    geometry: Optional[NeuronGeometry] = None,
    mode: str = "hybrid",
    k: int = 3,
    n_hubs: int = 1,
    intra_group_full: bool = True,
) -> Dict[str, List[str]]:
    """Build side-channel topology adjacency.

    Args:
        neurons: {nid: ResonanceNeuron}
        geometry: NeuronGeometry (auto-created + auto-assigned if None)
        mode: "full" | "knn" | "hub_spoke" | "hybrid"
        k: k-NN parameter (for knn mode); default 3
        n_hubs: number of hub neurons (for hub_spoke/hybrid mode)
        intra_group_full: hybrid mode — full mesh within (domain, spec) group?

    Returns:
        adjacency: {post_id: [pre_id, ...]} — post reads from each pre
    """
    nids = list(neurons.keys())
    n = len(nids)
    if n <= 1:
        return {nid: [] for nid in nids}

    if geometry is None:
        geometry = NeuronGeometry()
    _ensure_positions(geometry, nids)

    if mode == "full":
        return {post: [pre for pre in nids if pre != post] for post in nids}
    if mode == "knn":
        return _build_knn(nids, geometry, k=k)
    if mode == "hub_spoke":
        return _build_hub_spoke(nids, neurons, geometry, n_hubs=n_hubs)
    if mode == "hybrid":
        return _build_hybrid(
            nids, neurons, geometry, n_hubs=n_hubs, intra_group_full=intra_group_full,
        )
    raise ValueError(f"Unknown topology mode: {mode} (expected: full|knn|hub_spoke|hybrid)")


def _ensure_positions(geometry: NeuronGeometry, nids: List[str]) -> None:
    """Auto-assign geometry positions if any neuron is missing."""
    if set(nids) <= set(geometry.positions.keys()):
        return
    domain_to_nids: Dict[str, List[str]] = defaultdict(list)
    for nid in nids:
        domain_to_nids[_domain_of(nid)].append(nid)
    geometry.assign_domain_positions(dict(domain_to_nids))


def _build_knn(
    nids: List[str], geometry: NeuronGeometry, k: int = 3,
) -> Dict[str, List[str]]:
    """Symmetric k-nearest-neighbor topology.

    Each neuron connects to its k nearest neighbors. If A is in B's k-NN,
    B is also connected to A (symmetric), ensuring bidirectional signal flow.
    """
    k = min(k, len(nids) - 1)
    edges: Set[Tuple[str, str]] = set()
    for nid in nids:
        dists = sorted(
            ((geometry.distance(nid, other), other) for other in nids if other != nid),
            key=lambda x: x[0],
        )
        for _, other in dists[:k]:
            # Symmetric: add both directions
            edges.add((nid, other))
            edges.add((other, nid))
    adj: Dict[str, List[str]] = {nid: [] for nid in nids}
    for post, pre in edges:
        if pre not in adj[post]:
            adj[post].append(pre)
    return adj


def _select_hubs(
    nids: List[str],
    neurons: Dict[str, object],
    geometry: NeuronGeometry,
    n_hubs: int = 1,
) -> List[str]:
    """Select hub neurons: highest capacity, tiebreak by centroid proximity.

    Hub selection mirrors biology: larger-capacity neurons (association cortex)
    serve as integration hubs. Centroid tiebreak ensures hub is centrally located
    in geometry space (minimizing average distance to others).
    """
    if n_hubs >= len(nids):
        return list(nids)

    positions = [geometry.positions[nid] for nid in nids if nid in geometry.positions]
    if positions:
        centroid = torch.stack(positions).mean(dim=0)
    else:
        centroid = torch.zeros(geometry.embedding_dim)

    scored: List[Tuple[int, float, str]] = []
    for nid in nids:
        cap = _spec_capacity(neurons[nid])
        pos = geometry.positions.get(nid, centroid)
        dist = (pos - centroid).norm().item()
        # Sort key: -cap (higher capacity first), dist (closer to centroid first)
        scored.append((-cap, dist, nid))
    scored.sort()
    return [nid for _, _, nid in scored[:n_hubs]]


def _build_hub_spoke(
    nids: List[str],
    neurons: Dict[str, object],
    geometry: NeuronGeometry,
    n_hubs: int = 1,
) -> Dict[str, List[str]]:
    """Hub-spoke topology: hubs connect to all; non-hubs connect only to hubs.

    Hub reads from all spokes; spokes read from hub (and hub reads from them).
    Non-hub pairs do NOT connect directly — all traffic goes through hub.
    """
    hubs = set(_select_hubs(nids, neurons, geometry, n_hubs=n_hubs))
    adj: Dict[str, List[str]] = {nid: [] for nid in nids}
    for post in nids:
        for pre in nids:
            if pre == post:
                continue
            # Channel exists iff post is hub OR pre is hub
            if post in hubs or pre in hubs:
                adj[post].append(pre)
    return adj


def _build_hybrid(
    nids: List[str],
    neurons: Dict[str, object],
    geometry: NeuronGeometry,
    n_hubs: int = 1,
    intra_group_full: bool = True,
) -> Dict[str, List[str]]:
    """Hybrid topology — cortical hierarchy (default, highest upper limit).

    Three-tier connectivity mirroring cortical architecture:
      1. Same (domain, spec)   → full mesh (local microcircuit)
      2. Same domain, diff spec → via spec-hub (per-domain largest spec)
      3. Cross-domain           → via global-hub (globally largest spec)

    For the current 5-neuron all-zh case (4 compact + 1 standard):
      - (zh, compact) group: 4 compacts full mesh → 12 directed edges
      - Cross-spec within zh: compact↔standard via standard as spec-hub → 8 directed
      - Total: 20 directed (numerically same as full-mesh, but with explicit
        hub role for standard + distance-gated init_scale)

    The numerical equivalence for small same-domain ensembles is expected —
    the structural benefit appears when scaling to multi-domain ensembles
    (cross-domain traffic funnels through hub instead of N×(N-1) mesh).
    """
    global_hubs = set(_select_hubs(nids, neurons, geometry, n_hubs=n_hubs))

    # Per-domain spec-hub: highest-capacity neuron in each domain
    domain_to_spec_hub: Dict[str, str] = {}
    for nid in nids:
        domain = _domain_of(nid)
        if domain not in domain_to_spec_hub:
            domain_to_spec_hub[domain] = nid
        elif _spec_capacity(neurons[nid]) > _spec_capacity(neurons[domain_to_spec_hub[domain]]):
            domain_to_spec_hub[domain] = nid

    adj: Dict[str, List[str]] = {nid: [] for nid in nids}
    for post in nids:
        post_domain = _domain_of(post)
        post_spec = neurons[post].config.spec
        post_spec_hub = domain_to_spec_hub.get(post_domain, post)

        for pre in nids:
            if pre == post:
                continue
            pre_domain = _domain_of(pre)
            pre_spec = neurons[pre].config.spec

            # Tier 1: same (domain, spec) → full mesh
            if pre_domain == post_domain and pre_spec == post_spec:
                if intra_group_full:
                    adj[post].append(pre)
                continue

            # Tier 2: same domain, diff spec → via spec-hub
            if pre_domain == post_domain:
                if post == post_spec_hub or pre == post_spec_hub:
                    adj[post].append(pre)
                continue

            # Tier 3: cross-domain → via global-hub
            if post in global_hubs or pre in global_hubs:
                adj[post].append(pre)

    return adj


# ── Channel establishment ────────────────────────────────────────────────

def establish_topology_channels(
    neurons: Dict[str, object],
    topology: Dict[str, List[str]],
    geometry: Optional[NeuronGeometry] = None,
    channel_type: str = "excite",
    distance_gated_scale: bool = True,
) -> Dict[str, int]:
    """Establish side_channels according to topology adjacency.

    Replaces the nested-loop `for post: for pre: establish...` pattern with
    topology-driven establishment. Distance-gated init_scale gives near
    neighbors a stronger initial connection (biological prior: nearby neurons
    form stronger synapses).

    Args:
        neurons: {nid: ResonanceNeuron}
        topology: {post_id: [pre_id, ...]} from build_topology()
        geometry: for distance-gated init_scale (None → uniform 50.0)
        channel_type: "excite" or "inhibit"
        distance_gated_scale: if True, init_scale = 10 + 40 * distance_gate
            (near neighbor gate≈1 → 50.0, far gate≈0 → 10.0)

    Returns:
        stats: {nid: n_channels_established}
    """
    stats: Dict[str, int] = {}
    for post_id, pre_ids in topology.items():
        if post_id not in neurons:
            continue
        post_neuron = neurons[post_id]
        n = 0
        for pre_id in pre_ids:
            if pre_id not in neurons:
                continue
            pre_neuron = neurons[pre_id]

            if distance_gated_scale and geometry is not None:
                gate = geometry.distance_gate(post_id, pre_id)
                # near (gate≈1) → 50.0 (full default), far (gate≈0) → 10.0 (minimum)
                init_scale = 10.0 + 40.0 * gate
            else:
                init_scale = 50.0

            post_neuron.establish_side_channel(
                pre_id, pre_neuron,
                channel_type=channel_type,
                init_scale=init_scale,
            )
            n += 1
        stats[post_id] = n
    return stats


# ── Checkpoint inference ─────────────────────────────────────────────────

def infer_topology_from_state(side_channels_state: Dict) -> Dict[str, List[str]]:
    """Reconstruct topology adjacency from checkpoint's side_channels_state.

    Eval scripts call this to auto-match the topology used during training,
    without needing explicit topology metadata in the checkpoint. The channel
    keys in the checkpoint implicitly encode which (post, pre) pairs existed.

    Args:
        side_channels_state: {post_id: {"excite": {pre_id: ...}, "inhibit": {...}}}

    Returns:
        adjacency: {post_id: [pre_id, ...]}
    """
    adj: Dict[str, List[str]] = {}
    for post_id, channels in side_channels_state.items():
        pre_ids = list(channels.get("excite", {}).keys())
        adj[post_id] = pre_ids
    return adj


# ── Diagnostics ──────────────────────────────────────────────────────────

def topology_summary(topology: Dict[str, List[str]]) -> str:
    """Human-readable topology summary for logging."""
    n = len(topology)
    edges = sum(len(v) for v in topology.values())
    max_edges = n * (n - 1) if n > 1 else 0
    density = edges / max_edges if max_edges > 0 else 0.0
    avg_k = edges / n if n > 0 else 0.0
    return (f"neurons={n}, directed_edges={edges}, "
            f"density={density:.1%} of full-mesh, avg_indegree={avg_k:.1f}")


def topology_detail(
    topology: Dict[str, List[str]], neurons: Optional[Dict[str, object]] = None,
) -> str:
    """Detailed topology report with hub identification (for training logs)."""
    lines = [topology_summary(topology)]
    if neurons:
        # Identify hubs (highest indegree)
        indegree: Dict[str, int] = defaultdict(int)
        for pre_ids in topology.values():
            for pre in pre_ids:
                indegree[pre] += 1
        for nid in sorted(indegree, key=lambda x: -indegree[x]):
            spec = neurons[nid].config.spec if nid in neurons else "?"
            lines.append(f"  {nid} ({spec}): indegree={indegree[nid]}")
    return "\n".join(lines)
