import torch
from torch import Tensor
from torch.autograd import Function


def _safe_inv(s: Tensor, damping: float = 1e-12) -> Tensor:
    return s / (s * s + damping * damping)


def _pairwise_s2_diff(s: Tensor) -> Tensor:
    s2 = s * s
    return s2.unsqueeze(0) - s2.unsqueeze(1)


def _standard_F_from_diff(diff: Tensor) -> Tensor:
    eye = torch.eye(diff.shape[0], dtype=diff.dtype, device=diff.device)
    F = 1.0 / (diff + eye)
    F.diagonal().zero_()
    return F


def _geom_sum(r: Tensor, K: int) -> Tensor:
    denom = 1.0 - r
    num = 1.0 - r.pow(K)
    tiny = torch.finfo(r.dtype).eps
    return torch.where(denom.abs() < tiny, torch.full_like(r, float(K)), num / denom)


def _clusters_from_s(s: Tensor, tau: float):
    if tau <= 0.0:
        return []
    k = s.shape[0]
    clusters = []
    start = 0
    while start < k:
        center = s[start]
        end = start + 1
        while end < k and torch.abs(s[end] - center).item() < tau:
            end += 1
        if end - start > 1:
            clusters.append((start, end))
        start = end
    return clusters


def build_F_lorentzian(s: Tensor, **kwargs) -> Tensor:
    eps = float(kwargs.get("eps", 0.0))
    diff = _pairwise_s2_diff(s)
    if eps > 0.0:
        F = diff / (diff * diff + eps * eps)
    else:
        F = _standard_F_from_diff(diff)
    F.diagonal().zero_()
    return F


def build_F_freeze(s: Tensor, **kwargs) -> Tensor:
    tau = float(kwargs.get("tau", 0.0))
    diff = _pairwise_s2_diff(s)
    F = _standard_F_from_diff(diff)
    if tau > 0.0:
        F = torch.where(diff.abs() < tau, torch.zeros_like(F), F)
    F.diagonal().zero_()
    return F


def build_F_taylor(s: Tensor, **kwargs) -> Tensor:
    tau = float(kwargs.get("tau", 1e-8))
    K = int(kwargs.get("K", 8))
    s2 = s * s
    si2 = s2.unsqueeze(1)
    sj2 = s2.unsqueeze(0)
    diff = sj2 - si2
    F = _standard_F_from_diff(diff)
    if tau > 0.0:
        tiny = torch.finfo(s.dtype).eps
        si2_safe = torch.where(si2.abs() < tiny, torch.full_like(si2, tiny), si2)
        sj2_safe = torch.where(sj2.abs() < tiny, torch.full_like(sj2, tiny), sj2)
        r_up = si2_safe / sj2_safe
        r_dn = sj2_safe / si2_safe
        approx_up = _geom_sum(r_up, K) / sj2_safe
        approx_dn = -_geom_sum(r_dn, K) / si2_safe
        approx = torch.where(sj2_safe >= si2_safe, approx_up, approx_dn)
        near = diff.abs() < tau
        F = torch.where(near, approx, F)
    F.diagonal().zero_()
    return F


def build_F_degpert(s: Tensor, **kwargs) -> Tensor:
    tau = float(kwargs.get("tau", 1e-8))
    A = kwargs.get("A")
    U = kwargs.get("U")
    s2_eff = (s * s).clone()
    clusters = _clusters_from_s(s, tau)
    if A is not None and U is not None and len(clusters) > 0:
        AAT = A @ A.T
        for start, end in clusters:
            idx = torch.arange(start, end, device=s.device)
            Uc = U[:, idx]
            H = Uc.T @ AAT @ Uc
            eigvals = torch.linalg.eigvalsh(H).real.to(s.dtype)
            s2_eff[idx] = eigvals
    diff = s2_eff.unsqueeze(0) - s2_eff.unsqueeze(1)
    F = _standard_F_from_diff(diff)
    for start, end in clusters:
        idx = torch.arange(start, end, device=s.device)
        F[idx.unsqueeze(1), idx.unsqueeze(0)] = 0.0
    F.diagonal().zero_()
    return F


def build_F_method(s: Tensor, method: str = "lorentzian", **kwargs) -> Tensor:
    key = method.lower()
    if key == "lorentzian":
        return build_F_lorentzian(s, **kwargs)
    if key in {"freeze", "spectral_freeze", "spectral-freeze"}:
        return build_F_freeze(s, **kwargs)
    if key in {"taylor", "series", "geometric"}:
        return build_F_taylor(s, **kwargs)
    if key in {"degpert", "degenerate_perturbation", "degenerate-perturbation"}:
        return build_F_degpert(s, **kwargs)
    raise ValueError(f"Unknown F method: {method}")


def _solve_sylvester_colwise(
    A: Tensor,
    s2: Tensor,
    rhs: Tensor,
    U: Tensor,
    maxiter: int = 120,
    tol: float = 1e-10,
) -> Tensor:
    k = s2.shape[0]
    mu = 2.0 * s2.max().item()
    shifts = s2
    AAT = A @ A.T
    UUT = U @ U.T

    def matvec(V: Tensor) -> Tensor:
        return shifts.unsqueeze(0) * V - AAT @ V + mu * UUT @ V

    x = torch.zeros_like(rhs)
    r = rhs - matvec(x)
    p = r.clone()
    rr = (r * r).sum(dim=0)
    rhs_norm = rhs.norm(dim=0).clamp_min(1e-30)

    for _ in range(maxiter):
        Ap = matvec(p)
        pAp = (p * Ap).sum(dim=0).clamp_min(1e-30)
        alpha = rr / pAp
        x = x + alpha.unsqueeze(0) * p
        r = r - alpha.unsqueeze(0) * Ap
        rr_new = (r * r).sum(dim=0)
        rel_res = rr_new.sqrt() / rhs_norm
        if rel_res.max().item() < tol:
            break
        beta = rr_new / rr.clamp_min(1e-30)
        p = r + beta.unsqueeze(0) * p
        rr = rr_new

    return x


def build_F(s: Tensor, eps: float = 0.0) -> Tensor:
    return build_F_method(s, method="lorentzian", eps=eps)


def trunc_svd_forward(A: Tensor, k: int):
    U, s, Vh = torch.linalg.svd(A, full_matrices=False)
    return U[:, :k].contiguous(), s[:k].contiguous(), Vh[:k, :].contiguous()


def trunc_svd_jvp(A, U, s, Vh, dA, eps=0.0, f_method="lorentzian", **f_kwargs):
    k = U.shape[1]
    n, m = A.shape
    V = Vh.T
    s_diag = torch.diag(s)
    s_inv_diag = torch.diag(_safe_inv(s))
    build_kwargs = dict(f_kwargs)
    build_kwargs.setdefault("eps", eps)
    build_kwargs.setdefault("A", A)
    build_kwargs.setdefault("U", U)
    F = build_F_method(s, method=f_method, **build_kwargs)

    UtdAV = U.T @ dA @ V
    VtdAtU = UtdAV.T

    dS = torch.diag(UtdAV)
    dfU1 = F * (UtdAV @ s_diag + s_diag @ VtdAtU)
    dfV1 = F * (s_diag @ UtdAV + VtdAtU @ s_diag)

    dfU2 = torch.zeros_like(U)
    dfV2 = torch.zeros_like(V)

    if k < min(n, m):
        P_U = torch.eye(n, dtype=A.dtype, device=A.device) - U @ U.T
        P_V = torch.eye(m, dtype=A.dtype, device=A.device) - V @ V.T
        if n <= m:
            rhs = P_U @ dA @ V @ s_diag + A @ P_V @ dA.T @ U
            rhs = P_U @ rhs
            dfU2 = _solve_sylvester_colwise(A, s * s, rhs, U)
            dfU2 = P_U @ dfU2
            dfV2 = P_V @ dA.T @ U @ s_inv_diag + A.T @ dfU2 @ s_inv_diag
        else:
            rhs = P_V @ dA.T @ U @ s_diag + A.T @ P_U @ dA @ V
            rhs = P_V @ rhs
            dfV2 = _solve_sylvester_colwise(A.T, s * s, rhs, V)
            dfV2 = P_V @ dfV2
            dfU2 = P_U @ dA @ V @ s_inv_diag + A @ dfV2 @ s_inv_diag
    elif k == min(n, m):
        if n > m:
            P_U = torch.eye(n, dtype=A.dtype, device=A.device) - U @ U.T
            dfU2 = P_U @ dA @ V @ s_inv_diag
        elif n < m:
            P_V = torch.eye(m, dtype=A.dtype, device=A.device) - V @ V.T
            dfV2 = P_V @ dA.T @ U @ s_inv_diag

    dU = U @ dfU1 + dfU2
    dV = V @ dfV1 + dfV2
    return dU, dS, dV.T


def trunc_svd_vjp(
    A, U, s, Vh, grad_U, grad_s, grad_Vh, eps=0.0, f_method="lorentzian", **f_kwargs
):
    k = U.shape[1]
    n, m = A.shape
    V = Vh.T
    grad_V = grad_Vh.T
    s_diag = torch.diag(s)
    s_inv_diag = torch.diag(_safe_inv(s))
    build_kwargs = dict(f_kwargs)
    build_kwargs.setdefault("eps", eps)
    build_kwargs.setdefault("A", A)
    build_kwargs.setdefault("U", U)
    F = build_F_method(s, method=f_method, **build_kwargs)

    R = U.T @ grad_U
    Rt = V.T @ grad_V

    Omega = F * ((R - R.T) @ s_diag + s_diag @ (Rt - Rt.T))
    grad_A = U @ (torch.diag(grad_s) + Omega) @ Vh

    if k < min(n, m):
        P_U = torch.eye(n, dtype=A.dtype, device=A.device) - U @ U.T
        P_V = torch.eye(m, dtype=A.dtype, device=A.device) - V @ V.T
        P_U_gU = P_U @ grad_U
        P_V_gV = P_V @ grad_V
        if n <= m:
            X_bar = P_U @ (P_U_gU + A @ P_V_gV @ s_inv_diag)
            Z = _solve_sylvester_colwise(A, s * s, X_bar, U)
            Z = P_U @ Z
            grad_A = (
                grad_A + Z @ s_diag @ Vh + U @ Z.T @ A @ P_V + U @ s_inv_diag @ P_V_gV.T
            )
        else:
            Y_bar = P_V @ (P_V_gV + A.T @ P_U_gU @ s_inv_diag)
            W = _solve_sylvester_colwise(A.T, s * s, Y_bar, V)
            W = P_V @ W
            grad_A = (
                grad_A + U @ s_diag @ W.T + P_U @ A @ W @ Vh + P_U_gU @ s_inv_diag @ Vh
            )
    elif k == min(n, m):
        if n > m:
            P_U = torch.eye(n, dtype=A.dtype, device=A.device) - U @ U.T
            grad_A = grad_A + P_U @ grad_U @ s_inv_diag @ Vh
        elif n < m:
            P_V = torch.eye(m, dtype=A.dtype, device=A.device) - V @ V.T
            grad_A = grad_A + U @ s_inv_diag @ (P_V @ grad_V).T

    return grad_A


class TruncatedSVD(Function):
    @staticmethod
    def forward(ctx, A, k, eps=0.0):
        U, s, Vh = trunc_svd_forward(A, k)
        ctx.save_for_backward(U, s, Vh, A)
        ctx.eps = eps
        return U, s, Vh

    @staticmethod
    def backward(ctx, grad_U, grad_s, grad_Vh):
        U, s, Vh, A = ctx.saved_tensors
        grad_A = trunc_svd_vjp(
            A,
            U,
            s,
            Vh,
            grad_U if grad_U is not None else torch.zeros_like(U),
            grad_s if grad_s is not None else torch.zeros_like(s),
            grad_Vh if grad_Vh is not None else torch.zeros_like(Vh),
            eps=ctx.eps,
        )
        return grad_A, None, None


@torch.compile
def trunc_svd(A, k, eps=0.0):
    return TruncatedSVD.apply(A, k, eps)
