import argparse
import json
import sys
import time

import numpy as np
from safetensors.torch import load_file
import torch

from tsvd_real import trunc_svd_forward, trunc_svd_jvp, trunc_svd_vjp
from dobi_svd import stable_lowrank_SVD

METHODS = [
    ("Dobi-SVD", "dobi", "lorentzian", {"eps": 1e-4}),
    ("lorentzian eps=1e-4", "ours", "lorentzian", {"eps": 1e-4}),
    ("freeze tau=1e-6", "ours", "freeze", {"tau": 1e-6}),
    ("taylor tau=1e-8 K=16", "ours", "taylor", {"tau": 1e-8, "K": 16}),
    ("degpert tau=1e-6", "ours", "degpert", {"tau": 1e-6}),
    ("exact eps=0", "ours", "lorentzian", {"eps": 0.0}),
]


def _dobi_vjp(A, U, s, Vh, gU, gs, gVh):
    ctx = type("C", (), {"saved_tensors": (U, s, Vh.T)})()
    gA, _ = stable_lowrank_SVD.backward(ctx, gU, gs, gVh.T)
    return gA


def _evaluate(A, U, s, Vh, dA, gU, gs, gVh, impl, f_method, f_kwargs):
    dU, dS, dVh = trunc_svd_jvp(A, U, s, Vh, dA, f_method=f_method, **f_kwargs)
    if torch.isnan(dU).any() or torch.isnan(dS).any() or torch.isnan(dVh).any():
        return {"adj": float("nan"), "grad_norm": float("nan")}
    lhs = (gU * dU).sum() + (gs * dS).sum() + (gVh * dVh).sum()
    if impl == "dobi":
        gA = _dobi_vjp(A, U, s, Vh, gU, gs, gVh)
    else:
        gA = trunc_svd_vjp(A, U, s, Vh, gU, gs, gVh, f_method=f_method, **f_kwargs)
    if torch.isnan(gA).any():
        return {"adj": float("nan"), "grad_norm": gA.norm().item()}
    rhs = (gA * dA).sum()
    adj = ((lhs - rhs).abs() / (lhs.abs() + 1e-30)).item()
    return {"adj": adj, "grad_norm": gA.norm().item()}


def pick_ranks(s_full):
    n = len(s_full)
    gaps = (s_full[:-1] - s_full[1:]).numpy()
    energy = (s_full**2).cumsum(0) / (s_full**2).sum()
    energy = energy.numpy()

    ranks = {}
    min_gap_idx = int(np.argmin(gaps[1:])) + 1
    ranks["min_gap"] = min_gap_idx + 1

    for thresh, label in [(0.5, "energy_50"), (0.9, "energy_90"), (0.99, "energy_99")]:
        idx = int(np.searchsorted(energy, thresh))
        ranks[label] = max(2, min(idx + 1, n - 1))

    return ranks


def run_layer(name, W, rank_label, k):
    W = W.to(torch.float64)
    n, m = W.shape
    r = min(n, m)
    if k >= r or k < 2:
        return None

    _, s_full, _ = torch.linalg.svd(W, full_matrices=False)
    U, s, Vh = trunc_svd_forward(W, k)

    torch.manual_seed(hash(name) & 0xFFFFFFFF)
    dA = torch.randn_like(W)
    gU = torch.randn_like(U)
    gs = torch.randn(k, dtype=W.dtype)
    gVh = torch.randn_like(Vh)

    spectral = {
        "shape": [n, m],
        "k": k,
        "rank_label": rank_label,
        "s_k_minus_1": s_full[k - 1].item(),
        "s_k": s_full[k].item() if k < r else 0.0,
        "boundary_gap": (s_full[k - 1] - s_full[k]).item() if k < r else float("inf"),
        "boundary_ratio": (s_full[k - 1] / s_full[k]).item()
        if k < r and s_full[k] > 0
        else float("inf"),
        "s_min_kept": s_full[k - 1].item(),
        "s_max_kept": s_full[0].item(),
    }

    method_results = {}
    for label, impl, f_method, f_kwargs in METHODS:
        t0 = time.perf_counter()
        res = _evaluate(W, U, s, Vh, dA, gU, gs, gVh, impl, f_method, f_kwargs)
        res["ms"] = (time.perf_counter() - t0) * 1e3
        method_results[label] = res

    return {"spectral": spectral, "methods": method_results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=str, required=True, help="Path to model.safetensors"
    )
    args = parser.parse_args()

    

    weights = load_file(args.weights)

    mat_names = sorted(
        [
            n
            for n, t in weights.items()
            if t.ndim == 2 and min(t.shape) >= 64 and n.startswith("h.")
        ]
    )

    results = {}
    for name in mat_names:
        W = weights[name].to(torch.float64)
        _, s_full, _ = torch.linalg.svd(W, full_matrices=False)
        ranks = pick_ranks(s_full)
        for rank_label, k in ranks.items():
            key = f"{name}|{rank_label}|k={k}"
            res = run_layer(name, W, rank_label, k)
            if res is not None:
                results[key] = res

    json.dump(results, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
