#!/usr/bin/env python3
"""
Gossip learning with a Central Trust Aggregator (CTA) and a FederatedTrust-style
trust score, inspired by GRANITE (arXiv:2504.17471).

Architecture
    data plane    fully decentralised: local SGD + gossip of model *updates*
    control plane one central component that only sees trust observations
                  (never model weights) and broadcasts global trust scores.

What is simplified relative to the papers (be honest about this in the README):
    * HaPSLite : trust-weighted peer sampling. The GRANITE paper's HaPS keeps a
                 growing set of encountered identifiers with a proven
                 exponential decay of the Byzantine share. This does not.
    * APTLite  : norm threshold (multiplier x trimmed running median). The
                 paper's APT derives thresholds from a Chernoff bound.
    * GTS      : drop the update farthest from the centroid, average the rest.
    * Attack   : scaled sign-flip (-attack_scale x honest delta). This is not
                 the Fall-of-Empires attack, which scales against the honest mean.
    * Byzantine nodes do not submit trust observations (so trust manipulation by
      Byzantine observers is NOT evaluated).
    * Trust_FL: Privacy, Explainability and Accountability are derived from
      configuration (no DP / secure aggregation here), not measured.

Usage
    python granite.py                         # real MNIST, 50 rounds, seed 42
    python granite.py --seeds 0 1 2 3 4       # several seeds, mean/std summary
    python granite.py --aggregator eigentrust # weighted | median | eigentrust
    python granite.py --synthetic --rounds 5  # offline smoke test, no download

Colab (upload granite.py to the session, or put it in Drive), then in a cell:
    %run granite.py --synthetic --rounds 5          # smoke test
    %run granite.py --rounds 50 --seeds 0 1 2       # real MNIST
or from Python:
    import granite
    res = granite.run(rounds=50, seeds=[0, 1, 2], aggregator="weighted")
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def in_notebook() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


# ================================================================
# Config
# ================================================================
@dataclass
class ExperimentConfig:
    total_nodes: int = 10
    num_byzantine: int = 1
    view_size: int = 5
    num_rounds: int = 50
    learning_rate: float = 0.01
    warm_after: int = 3            # rounds of uniform sampling before trust-weighting
    train_per_node: int = 100
    test_samples: int = 1000
    seed: int = 42
    aggregator: str = "weighted"   # weighted | median | eigentrust
    attack_scale: float = 5.0      # Byzantine sends -attack_scale * honest delta
    apt_multiplier: float = 3.0
    apt_window: int = 5
    apt_min_keep: float = 0.5      # APT never discards more than half the view
    cta_decay: float = 0.85
    cta_flag_penalty: float = 0.15
    global_blend: float = 0.7      # tau_eff = 0.7 * global + 0.3 * local
    synthetic: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ================================================================
# Model + data
# ================================================================
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        return self.fc2(F.relu(self.fc1(x)))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_synthetic(n: int, seed: int):
    """Class-prototype + noise images. Only for offline smoke tests."""
    g = torch.Generator().manual_seed(seed)
    protos = torch.randn(10, 1, 28, 28, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    X = protos[y] + 0.8 * torch.randn(n, 1, 28, 28, generator=g)
    return X, y


def load_data(cfg: ExperimentConfig):
    """Random (seeded) train subset split equally across nodes; random test subset."""
    rng = np.random.RandomState(cfg.seed)
    n_train = cfg.train_per_node * cfg.total_nodes
    if cfg.synthetic:
        X, y = make_synthetic(n_train + cfg.test_samples, cfg.seed)
        Xtr, ytr, Xte, yte = X[:n_train], y[:n_train], X[n_train:], y[n_train:]
    else:
        from torchvision import datasets, transforms
        tf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))])
        tr = datasets.MNIST("./data", train=True, download=True, transform=tf)
        te = datasets.MNIST("./data", train=False, download=True, transform=tf)
        tri = rng.choice(len(tr), n_train, replace=False)
        tei = rng.choice(len(te), cfg.test_samples, replace=False)
        Xtr = torch.stack([tr[int(i)][0] for i in tri])
        ytr = torch.tensor([tr[int(i)][1] for i in tri])
        Xte = torch.stack([te[int(i)][0] for i in tei])
        yte = torch.tensor([te[int(i)][1] for i in tei])
    parts = np.array_split(rng.permutation(n_train), cfg.total_nodes)
    loaders = [
        DataLoader(TensorDataset(Xtr[torch.as_tensor(p)], ytr[torch.as_tensor(p)]),
                   batch_size=32, shuffle=True,
                   generator=torch.Generator().manual_seed(cfg.seed + i))
        for i, p in enumerate(parts)
    ]
    return loaders, (Xte, yte)


# ================================================================
# FederatedTrust-style evaluator
# ================================================================
@dataclass
class TrustWeights:
    privacy: float = 2.0
    robustness: float = 2.0
    fairness: float = 1.5
    explainability: float = 1.0
    accountability: float = 1.5
    federation: float = 1.0

    def as_array(self) -> np.ndarray:
        w = np.array([self.privacy, self.robustness, self.fairness,
                      self.explainability, self.accountability, self.federation])
        return w / w.sum()


class TrustEvaluator:
    """Six-pillar score. Robustness, Fairness and Federation are measured from the
    run; Privacy, Explainability and Accountability come from configuration."""

    def __init__(self, model_params: int, factsheet: Dict,
                 use_dp: bool = False, dp_epsilon: Optional[float] = None,
                 secure_aggregation: bool = False,
                 weights: Optional[TrustWeights] = None):
        self.model_params = model_params
        self.factsheet = factsheet
        self.use_dp = use_dp
        self.dp_epsilon = dp_epsilon
        self.secure_aggregation = secure_aggregation
        self.weights = weights or TrustWeights()

    def _privacy(self) -> float:
        score = 0.10
        if self.use_dp:
            eps = self.dp_epsilon
            score += 0.40 if eps is None else 0.55 * float(np.clip((10.0 - eps) / 9.0, 0, 1))
        if self.secure_aggregation:
            score += 0.35
        return min(score, 1.0)

    @staticmethod
    def _robustness(accuracy: float, byz_global_trust: float, poison_pass_rate: float) -> float:
        """Mean of: accuracy, Byzantine suppression (1 - CTA trust in Byzantine
        nodes), and 1 - fraction of Byzantine updates that passed APT."""
        return float(np.mean([
            np.clip(accuracy, 0, 1),
            np.clip(1.0 - byz_global_trust, 0, 1),
            np.clip(1.0 - poison_pass_rate, 0, 1),
        ]))

    @staticmethod
    def _fairness(participation: List[float], outcome: Optional[List[float]]) -> float:
        def cv_score(xs):
            a = np.asarray(xs, dtype=float)
            if len(a) < 2 or a.mean() <= 0:
                return 0.0
            return float(np.clip(1.0 - a.std() / a.mean(), 0, 1))
        parts = [cv_score(participation)]
        if outcome is not None and len(outcome) >= 2:
            parts.append(cv_score(outcome))
        return float(np.mean(parts))

    def _explainability(self) -> float:
        p = self.model_params
        if p <= 1_000:
            return 0.90
        if p <= 50_000:
            return 0.70
        if p <= 500_000:
            return 0.45
        if p <= 10_000_000:
            return 0.25
        return 0.10

    def _accountability(self) -> float:
        weighted = {"project_specification": 2, "data": 2,
                    "participants": 1, "configuration": 1, "system": 1}
        earned = sum(w for k, w in weighted.items() if k in self.factsheet)
        return earned / sum(weighted.values())

    @staticmethod
    def _federation(inter_node_variance: float) -> float:
        """Log-scaled inter-node spread, E_params[Var_nodes(theta)]. Note: with
        small models this term saturates near 1 early; treat it as weak evidence."""
        conv = float(np.clip(1.0 - np.log10(inter_node_variance * 1000.0 + 1.0) / 3.0, 0, 1))
        stability = 0.05 if inter_node_variance < 1e-3 else 0.0
        return min(conv + stability, 1.0)

    def evaluate(self, s: Dict) -> Dict[str, float]:
        dims = {
            "Privacy": self._privacy(),
            "Robustness": self._robustness(s["accuracy"], s["byz_global_trust"],
                                           s["poison_pass_rate"]),
            "Fairness": self._fairness(s["sampling_counts"], s.get("outcome_accuracy")),
            "Explainability": self._explainability(),
            "Accountability": self._accountability(),
            "Federation": self._federation(s["variance"]),
        }
        dims["Trust_FL_Score"] = float(np.dot(self.weights.as_array(),
                                              [dims[k] for k in list(dims)[:6]]))
        return dims


# ================================================================
# Local trust observation + Central Trust Aggregator
# ================================================================
@dataclass
class TrustObservation:
    round: int
    observer: int
    target: int
    cosine: float      # cosine to the observer's robust centroid
    norm: float        # L2 norm of the target's update
    flagged: bool      # filtered by the observer's APT

    @property
    def score(self) -> float:
        return float(max(self.cosine, 0.0))


class TrustAggregator(ABC):
    """Sees observations only. Never sees model weights."""

    def __init__(self, n: int, decay: float, flag_penalty: float):
        self.n, self.decay, self.flag_penalty = n, decay, flag_penalty
        self.base = np.full(n, 0.5)    # neutral prior before any observation
        self.flags = np.zeros(n)       # exponentially decayed flag counts
        self.trust = self.base.copy()

    def update(self, observations: List[TrustObservation]) -> None:
        self.flags *= self.decay
        for o in observations:
            self.flags[o.target] += float(o.flagged)
        self._update_base(observations)
        self.trust = np.clip(self.base - self.flag_penalty * self.flags, 0.0, 1.0)

    @abstractmethod
    def _update_base(self, observations: List[TrustObservation]) -> None: ...

    def global_trust(self) -> np.ndarray:
        return self.trust.copy()


class WeightedAverageAggregator(TrustAggregator):
    """Decayed mean of observers' scores, each weighted by the observer's own trust."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.num = np.zeros(self.n)
        self.den = np.zeros(self.n)

    def _update_base(self, observations):
        self.num *= self.decay
        self.den *= self.decay
        for o in observations:
            w = max(self.trust[o.observer], 1e-3)
            self.num[o.target] += w * o.score
            self.den[o.target] += w
        seen = self.den > 1e-9
        self.base = np.where(seen, self.num / np.maximum(self.den, 1e-9), 0.5)


class MedianAggregator(TrustAggregator):
    """Trimmed median of this round's scores per target, smoothed with an EMA."""

    def _update_base(self, observations):
        by_target: Dict[int, List[float]] = {}
        for o in observations:
            by_target.setdefault(o.target, []).append(o.score)
        for t, sc in by_target.items():
            sc = sorted(sc)
            if len(sc) > 4:
                sc = sc[1:-1]
            self.base[t] = self.decay * self.base[t] + (1 - self.decay) * float(np.median(sc))


class EigenTrustAggregator(TrustAggregator):
    """EigenTrust-style power iteration over an EMA'd observer x target score matrix."""

    def __init__(self, *a, alpha: float = 0.1, iters: int = 50, **k):
        super().__init__(*a, **k)
        self.S = np.zeros((self.n, self.n))
        self.alpha, self.iters = alpha, iters

    def _update_base(self, observations):
        for o in observations:
            self.S[o.observer, o.target] = (self.decay * self.S[o.observer, o.target]
                                            + (1 - self.decay) * o.score)
        rows = self.S.sum(axis=1, keepdims=True)
        C = np.where(rows > 1e-12, self.S / np.maximum(rows, 1e-12), 1.0 / self.n)
        p = np.full(self.n, 1.0 / self.n)
        t = p.copy()
        for _ in range(self.iters):
            t = (1 - self.alpha) * C.T @ t + self.alpha * p
        self.base = t / t.max()


AGGREGATORS = {"weighted": WeightedAverageAggregator,
               "median": MedianAggregator,
               "eigentrust": EigenTrustAggregator}


# ================================================================
# Per-node components
# ================================================================
class HaPSLite:
    """Uniform sampling during warm-up, then trust-weighted sampling using
    tau_eff = blend * global + (1 - blend) * local."""

    def __init__(self, node_id: int, all_ids: List[int], cfg: ExperimentConfig,
                 rng: np.random.RandomState):
        self.peers = [i for i in all_ids if i != node_id]
        self.cfg, self.rng = cfg, rng
        self.local = {i: 1.0 for i in self.peers}

    def effective(self, j: int, global_trust: np.ndarray) -> float:
        b = self.cfg.global_blend
        return b * float(global_trust[j]) + (1 - b) * self.local[j]

    def select_view(self, round_num: int, global_trust: np.ndarray) -> List[int]:
        k = min(self.cfg.view_size, len(self.peers))
        if round_num < self.cfg.warm_after:
            return [int(x) for x in self.rng.choice(self.peers, k, replace=False)]
        w = np.array([max(self.effective(j, global_trust), 1e-6) for j in self.peers])
        return [int(x) for x in self.rng.choice(self.peers, k, replace=False, p=w / w.sum())]

    def update_local(self, peer: int, cosine: float) -> None:
        if cosine < 0.0:
            self.local[peer] = max(self.local[peer] - 0.2, 0.0)
        elif cosine > 0.5:
            self.local[peer] = min(self.local[peer] + 0.1, 1.0)


class APTLite:
    """Flags updates whose L2 norm exceeds multiplier x (running mean of per-round
    trimmed medians). Never discards more than (1 - min_keep) of the view."""

    def __init__(self, multiplier: float, window: int, min_keep: float):
        self.multiplier, self.window, self.min_keep = multiplier, window, min_keep
        self.history: List[float] = []

    def update_baseline(self, norms: np.ndarray) -> None:
        s = sorted(norms.tolist())
        if len(s) > 4:
            trimmed = s[1:-1]          # drop lowest and highest
        elif len(s) > 2:
            trimmed = s[:-1]           # drop highest
        else:
            trimmed = s
        self.history.append(float(np.median(trimmed)))
        self.history = self.history[-self.window:]

    def keep_mask(self, norms: np.ndarray) -> np.ndarray:
        threshold = float(np.mean(self.history)) * self.multiplier
        keep = norms <= threshold
        min_keep = max(1, int(len(norms) * self.min_keep))
        if keep.sum() < min_keep:      # safety floor: keep the smallest-norm updates
            keep = np.zeros(len(norms), dtype=bool)
            keep[np.argsort(norms)[:min_keep]] = True
        return keep


def gts_aggregate(updates: np.ndarray) -> np.ndarray:
    """Simplified geometric trimmed summation: drop the update farthest from the
    centroid, average the rest."""
    if len(updates) <= 2:
        return updates.mean(axis=0)
    centroid = updates.mean(axis=0)
    far = int(np.argmax(np.linalg.norm(updates - centroid, axis=1)))
    return np.delete(updates, far, axis=0).mean(axis=0)


class GraniteNode:
    def __init__(self, node_id: int, loader: DataLoader, is_byzantine: bool,
                 cfg: ExperimentConfig, init_vec: torch.Tensor, all_ids: List[int]):
        self.id, self.loader, self.is_byzantine, self.cfg = node_id, loader, is_byzantine, cfg
        self.model = Net().to(DEVICE)
        vector_to_parameters(init_vec.clone().to(DEVICE), self.model.parameters())
        rng = np.random.RandomState(cfg.seed * 1000 + node_id)
        self.haps = HaPSLite(node_id, all_ids, cfg, rng)
        self.apt = APTLite(cfg.apt_multiplier, cfg.apt_window, cfg.apt_min_keep)
        self.start: np.ndarray
        self.delta: np.ndarray

    def get_vec(self) -> np.ndarray:
        return parameters_to_vector(self.model.parameters()).detach().cpu().numpy()

    def set_vec(self, v: np.ndarray) -> None:
        vector_to_parameters(torch.from_numpy(v.astype(np.float32)).to(DEVICE),
                             self.model.parameters())

    def begin_round(self) -> np.ndarray:
        """Train locally from the current (post-aggregation) weights and return the
        update this node broadcasts. Byzantine nodes broadcast a sign-flipped one."""
        self.start = self.get_vec()
        opt = torch.optim.SGD(self.model.parameters(), lr=self.cfg.learning_rate, momentum=0.9)
        self.model.train()
        for x, y in self.loader:
            opt.zero_grad()
            F.cross_entropy(self.model(x.to(DEVICE)), y.to(DEVICE)).backward()
            opt.step()
        self.delta = self.get_vec() - self.start
        return -self.cfg.attack_scale * self.delta if self.is_byzantine else self.delta

    def aggregate(self, peer_ids: List[int], peer_updates: List[np.ndarray],
                  round_num: int):
        """Honest nodes only. Returns (observations, flagged peer ids)."""
        U = np.stack(peer_updates)
        norms = np.linalg.norm(U, axis=1)
        self.apt.update_baseline(norms)

        # robust reference: centroid excluding the single largest-norm update
        ref = np.delete(U, int(np.argmax(norms)), axis=0).mean(axis=0) if len(U) > 2 \
            else U.mean(axis=0)
        ref_norm = float(np.linalg.norm(ref))

        keep = self.apt.keep_mask(norms)
        observations, flagged = [], []
        for u, pid, n, kept in zip(U, peer_ids, norms, keep):
            cos = float(u @ ref / (n * ref_norm)) if n > 1e-10 and ref_norm > 1e-10 else 1.0
            self.haps.update_local(pid, cos)
            if not kept:
                flagged.append(pid)
            observations.append(TrustObservation(round_num, self.id, pid, cos,
                                                 float(n), not kept))

        agg = gts_aggregate(U[keep])
        self.set_vec(self.start + 0.5 * (self.delta + agg))
        return observations, flagged


# ================================================================
# Simulation
# ================================================================
def accuracy_of(vec: np.ndarray, net: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    vector_to_parameters(torch.from_numpy(vec.astype(np.float32)).to(DEVICE), net.parameters())
    net.eval()
    with torch.no_grad():
        pred = net(X.to(DEVICE)).argmax(1)
    return float((pred.cpu() == y).float().mean())


def build_factsheet(cfg: ExperimentConfig, n_params: int) -> Dict:
    return {
        "project_specification": {"purpose": "Byzantine-resilient gossip learning with a "
                                             "central trust aggregator"},
        "participants": [{"name": f"Node_{i}"} for i in range(cfg.total_nodes)],
        "data": {"provenance": "synthetic" if cfg.synthetic else "MNIST (LeCun et al., 1998)",
                 "pre_processing": "mean 0.1307, std 0.3081"},
        "configuration": {"model": "CNN (Net)", "aggregation": "GTS (simplified)",
                          "rounds": cfg.num_rounds, "view_size": cfg.view_size},
        "system": {"model_size_bytes": n_params * 4},
    }


def run_experiment(cfg: ExperimentConfig, verbose: bool = True) -> Dict:
    set_seed(cfg.seed)
    loaders, (Xte, yte) = load_data(cfg)
    init_vec = parameters_to_vector(Net().parameters()).detach().cpu()  # shared init
    n_params = count_params(Net())

    ids = list(range(cfg.total_nodes))
    byz = set(range(cfg.num_byzantine))
    honest = [i for i in ids if i not in byz]
    nodes = {i: GraniteNode(i, loaders[i], i in byz, cfg, init_vec, ids) for i in ids}
    cta = AGGREGATORS[cfg.aggregator](cfg.total_nodes, cfg.cta_decay, cfg.cta_flag_penalty)
    evaluator = TrustEvaluator(n_params, build_factsheet(cfg, n_params))
    eval_net = Net().to(DEVICE)

    if verbose:
        print(f"GRANITE | {cfg.total_nodes} nodes | {cfg.num_byzantine} Byzantine | "
              f"{cfg.num_rounds} rounds | CTA={cfg.aggregator} | seed={cfg.seed} | "
              f"params={n_params:,} | device={DEVICE}")

    history = []
    for r in range(cfg.num_rounds):
        global_trust = cta.global_trust()
        updates = {i: nodes[i].begin_round() for i in ids}

        sampled = np.zeros(cfg.total_nodes)
        byz_slots = byz_flagged = hon_slots = hon_flagged = 0
        all_obs: List[TrustObservation] = []
        for i in honest:
            view = nodes[i].haps.select_view(r, global_trust)
            for pid in view:
                sampled[pid] += 1
            obs, flagged = nodes[i].aggregate(view, [updates[p] for p in view], r)
            all_obs.extend(obs)
            for pid in view:
                is_flag = pid in flagged
                if pid in byz:
                    byz_slots += 1
                    byz_flagged += is_flag
                else:
                    hon_slots += 1
                    hon_flagged += is_flag

        cta.update(all_obs)                         # control plane: observations only
        gt = cta.global_trust()

        vecs = np.stack([nodes[i].get_vec() for i in honest])
        accuracy = accuracy_of(vecs.mean(axis=0), eval_net, Xte, yte)
        per_node_acc = [accuracy_of(v, eval_net, Xte, yte) for v in vecs]
        variance = float(np.var(vecs, axis=0).mean())   # inter-node spread

        total_slots = max(byz_slots + hon_slots, 1)
        byz_global = float(np.mean([gt[b] for b in byz])) if byz else 0.0
        byz_local = float(np.mean([nodes[i].haps.local[b] for i in honest for b in byz])) \
            if byz else 0.0
        poison_pass = 1.0 - byz_flagged / byz_slots if byz_slots else 0.0

        tm = evaluator.evaluate({
            "accuracy": accuracy, "byz_global_trust": byz_global,
            "poison_pass_rate": poison_pass, "variance": variance,
            "sampling_counts": [sampled[i] for i in honest],  # honest participation only
            "outcome_accuracy": per_node_acc,
        })
        row = {
            "round": r + 1, "accuracy": accuracy,
            "min_node_accuracy": float(min(per_node_acc)),
            "byz_global_trust": byz_global, "byz_local_trust": byz_local,
            "byz_sampling_share": byz_slots / total_slots,
            "apt_detection_rate": (byz_flagged / byz_slots) if byz_slots else None,
            "apt_false_positive_rate": hon_flagged / max(hon_slots, 1),
            "inter_node_variance": variance, **tm,
        }
        history.append(row)
        if verbose:
            det = "n/a" if row["apt_detection_rate"] is None else f"{row['apt_detection_rate']:.2f}"
            print(f"Round {r + 1:3d} | acc={accuracy:.3f} | byz_cta={byz_global:.3f} | "
                  f"byz_share={row['byz_sampling_share']:.2f} | apt_det={det} | "
                  f"apt_fp={row['apt_false_positive_rate']:.2f} | "
                  f"Trust_FL={row['Trust_FL_Score']:.3f}")

    isolation = next((h["round"] for h in history if h["byz_global_trust"] < 0.1), None) \
        if byz else None
    best = max(history, key=lambda h: h["accuracy"])
    summary = {
        "final_accuracy": history[-1]["accuracy"],
        "peak_accuracy": best["accuracy"], "peak_round": best["round"],
        "isolation_round": isolation,   # first round CTA trust in Byzantine < 0.1
        "final_byz_global_trust": history[-1]["byz_global_trust"],
        "final_trust_fl": history[-1]["Trust_FL_Score"],
    }
    return {"config": asdict(cfg), "model_params": n_params,
            "summary": summary, "rounds": history}


# ================================================================
# Output
# ================================================================
def save_results(res: Dict, out_dir: str) -> str:
    seed = res["config"]["seed"]
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"{res['config']['aggregator']}_seed{seed}")
    with open(stem + ".json", "w") as f:
        json.dump(res, f, indent=2)
    with open(stem + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res["rounds"][0].keys()))
        w.writeheader()
        w.writerows(res["rounds"])
    return stem


def plot_results(res: Dict, path: str) -> None:
    import matplotlib
    if not in_notebook():
        matplotlib.use("Agg")      # headless scripts; keep the inline backend in Colab
    import matplotlib.pyplot as plt

    h = res["rounds"]
    r = [x["round"] for x in h]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax[0, 0].plot(r, [x["accuracy"] for x in h], label="mean honest model")
    ax[0, 0].plot(r, [x["min_node_accuracy"] for x in h], "--", label="worst honest node")
    ax[0, 0].set(title="Test accuracy", xlabel="round", ylim=(0, 1))
    ax[0, 0].legend()
    ax[0, 1].plot(r, [x["byz_global_trust"] for x in h], label="CTA global")
    ax[0, 1].plot(r, [x["byz_local_trust"] for x in h], "--", label="mean local (HaPS)")
    ax[0, 1].set(title="Trust in Byzantine node(s)", xlabel="round", ylim=(-0.05, 1.05))
    ax[0, 1].legend()
    ax[0, 2].plot(r, [x["byz_sampling_share"] for x in h])
    ax[0, 2].set(title="Byzantine share of honest views", xlabel="round", ylim=(0, 1))
    det = [np.nan if x["apt_detection_rate"] is None else x["apt_detection_rate"] for x in h]
    ax[1, 0].plot(r, det, label="detection (Byzantine flagged)")
    ax[1, 0].plot(r, [x["apt_false_positive_rate"] for x in h], "--",
                  label="false positives (honest flagged)")
    ax[1, 0].set(title="APT", xlabel="round", ylim=(-0.05, 1.05))
    ax[1, 0].legend()
    ax[1, 1].plot(r, [x["Trust_FL_Score"] for x in h])
    ax[1, 1].set(title="Trust_FL score", xlabel="round", ylim=(0, 1))
    pillars = ["Privacy", "Robustness", "Fairness", "Explainability", "Accountability", "Federation"]
    ax[1, 2].bar([p[:5] for p in pillars], [h[-1][p] for p in pillars])
    ax[1, 2].set(title=f"Pillars (final round {r[-1]})", ylim=(0, 1.05))
    fig.suptitle(f"CTA={res['config']['aggregator']}  seed={res['config']['seed']}  "
                 f"byzantine={res['config']['num_byzantine']}/{res['config']['total_nodes']}")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def print_summary(results: List[Dict]) -> None:
    def ms(key):
        v = [r["summary"][key] for r in results if r["summary"][key] is not None]
        return f"{np.mean(v):.3f} +/- {np.std(v):.3f} (n={len(v)})" if v else "n/a"
    print(f"\n=== Summary over {len(results)} seed(s) ===")
    for key in ("final_accuracy", "peak_accuracy", "isolation_round",
                "final_byz_global_trust", "final_trust_fl"):
        print(f"{key:24s} {ms(key)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    d = ExperimentConfig()
    ap.add_argument("--rounds", type=int, default=d.num_rounds)
    ap.add_argument("--nodes", type=int, default=d.total_nodes)
    ap.add_argument("--byzantine", type=int, default=d.num_byzantine)
    ap.add_argument("--view-size", type=int, default=d.view_size)
    ap.add_argument("--aggregator", choices=sorted(AGGREGATORS), default=d.aggregator)
    ap.add_argument("--seeds", type=int, nargs="+", default=[d.seed])
    ap.add_argument("--train-per-node", type=int, default=d.train_per_node)
    ap.add_argument("--test-samples", type=int, default=d.test_samples)
    ap.add_argument("--synthetic", action="store_true", help="offline smoke test data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-plots", action="store_true")
    a, _ = ap.parse_known_args()   # ignore the extra args Jupyter/Colab injects
    run_from_args(a)


def run_from_args(a) -> List[Dict]:
    results = []
    for seed in a.seeds:
        cfg = ExperimentConfig(total_nodes=a.nodes, num_byzantine=a.byzantine,
                               view_size=a.view_size, num_rounds=a.rounds, seed=seed,
                               aggregator=a.aggregator, train_per_node=a.train_per_node,
                               test_samples=a.test_samples, synthetic=a.synthetic)
        res = run_experiment(cfg)
        stem = save_results(res, a.out)
        if not a.no_plots:
            plot_results(res, stem + ".png")
            if in_notebook():
                from IPython.display import Image, display
                display(Image(stem + ".png"))
        results.append(res)
    print_summary(results)
    return results


def run(rounds: int = 50, nodes: int = 10, byzantine: int = 1, view_size: int = 5,
        aggregator: str = "weighted", seeds=(42,), train_per_node: int = 100,
        test_samples: int = 1000, synthetic: bool = False, out: str = "results",
        no_plots: bool = False) -> List[Dict]:
    """Notebook-friendly entry point (no argparse)."""
    a = argparse.Namespace(rounds=rounds, nodes=nodes, byzantine=byzantine,
                           view_size=view_size, aggregator=aggregator, seeds=list(seeds),
                           train_per_node=train_per_node, test_samples=test_samples,
                           synthetic=synthetic, out=out, no_plots=no_plots)
    return run_from_args(a)


if __name__ == "__main__":
    main()