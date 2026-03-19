import torch
from torch import Tensor
from torch.autograd import Function


def _safe_inv(s: Tensor, damping: float = 1e-12) -> Tensor:
    return s / (s * s + damping * damping)


def _solve_sylvester_colwise(AAH, s2, rhs, U):
    mu = 2.0 * s2.max().item()
    C = AAH - mu * (U @ U.mH)
    n = rhs.shape[0]
    k = s2.shape[0]
    I_n = torch.eye(n, dtype=rhs.dtype, device=rhs.device)
    A_batch = s2.to(rhs.dtype).reshape(k, 1, 1) * I_n.unsqueeze(0) - C.unsqueeze(0)
    x_batch = torch.linalg.solve(A_batch, rhs.T.unsqueeze(2))
    return x_batch.squeeze(2).T


def build_F(s: Tensor, eps: float = 0.0) -> Tensor:
    s2 = s * s
    diff = s2.unsqueeze(0) - s2.unsqueeze(1)
    if eps > 0.0:
        F = diff / (diff * diff + eps * eps)
    else:
        safe = diff + torch.eye(s.shape[0], dtype=s.real.dtype, device=s.device)
        F = 1.0 / safe
    F.diagonal().zero_()
    return F


def trunc_svd_forward(A: Tensor, k: int):
    U, s, Vh = torch.linalg.svd(A, full_matrices=False)
    return U[:, :k].contiguous(), s[:k].contiguous(), Vh[:k, :].contiguous()


def trunc_svd_jvp(A, U, s, Vh, dA, eps=0.0):
    k = U.shape[1]
    n, m = A.shape
    V = Vh.mH
    s_diag = torch.diag(s.to(A.dtype))
    s_inv_diag = torch.diag(_safe_inv(s).to(A.dtype))
    F = build_F(s, eps)

    P = U.mH @ dA @ V
    Ph = P.mH

    dS = torch.diag(P).real
    dfU1 = F * (P @ s_diag + s_diag @ Ph)
    dfV1 = F * (s_diag @ P + Ph @ s_diag)

    s_inv_real = _safe_inv(s)
    diag_gauge = 0.25 * s_inv_real * torch.diag(P - Ph)
    dfU1 = dfU1 + torch.diag(diag_gauge)
    dfV1 = dfV1 + torch.diag(diag_gauge)

    dfU2 = torch.zeros_like(U)
    dfV2 = torch.zeros_like(V)

    if k < min(n, m):
        P_U = torch.eye(n, dtype=A.dtype, device=A.device) - U @ U.mH
        P_V = torch.eye(m, dtype=A.dtype, device=A.device) - V @ V.mH
        if n <= m:
            rhs = P_U @ dA @ V @ s_diag + A @ P_V @ dA.mH @ U
            rhs = P_U @ rhs
            dfU2 = _solve_sylvester_colwise(A @ A.mH, s * s, rhs, U)
            dfU2 = P_U @ dfU2
            dfV2 = P_V @ dA.mH @ U @ s_inv_diag + A.mH @ dfU2 @ s_inv_diag
        else:
            rhs = P_V @ dA.mH @ U @ s_diag + A.mH @ P_U @ dA @ V
            rhs = P_V @ rhs
            dfV2 = _solve_sylvester_colwise(A.mH @ A, s * s, rhs, V)
            dfV2 = P_V @ dfV2
            dfU2 = P_U @ dA @ V @ s_inv_diag + A @ dfV2 @ s_inv_diag
    elif k == min(n, m):
        if n > m:
            P_U = torch.eye(n, dtype=A.dtype, device=A.device) - U @ U.mH
            dfU2 = P_U @ dA @ V @ s_inv_diag
        elif n < m:
            P_V = torch.eye(m, dtype=A.dtype, device=A.device) - V @ V.mH
            dfV2 = P_V @ dA.mH @ U @ s_inv_diag

    dU = U @ dfU1 + dfU2
    dV = V @ dfV1 + dfV2
    return dU, dS, dV.mH


def trunc_svd_vjp(A, U, s, Vh, grad_U, grad_s, grad_Vh, eps=0.0):
    k = U.shape[1]
    n, m = A.shape
    V = Vh.mH
    grad_V = grad_Vh.mH
    s_diag = torch.diag(s.to(A.dtype))
    s_inv_diag = torch.diag(_safe_inv(s).to(A.dtype))
    F = build_F(s, eps)

    R = U.mH @ grad_U
    Rt = V.mH @ grad_V

    Omega = F * ((R - R.mH) @ s_diag + s_diag @ (Rt - Rt.mH))

    D = torch.diag(0.25 * _safe_inv(s) * torch.diag(R))
    Dt = torch.diag(0.25 * _safe_inv(s) * torch.diag(Rt))
    gauge = (D - D.mH) + (Dt - Dt.mH)

    grad_A = U @ (torch.diag(grad_s.to(A.dtype)) + Omega + gauge) @ Vh

    if k < min(n, m):
        P_U = torch.eye(n, dtype=A.dtype, device=A.device) - U @ U.mH
        P_V = torch.eye(m, dtype=A.dtype, device=A.device) - V @ V.mH
        P_U_gU = P_U @ grad_U
        P_V_gV = P_V @ grad_V
        if n <= m:
            X_bar = P_U @ (P_U_gU + A @ P_V_gV @ s_inv_diag)
            Z = _solve_sylvester_colwise(A @ A.mH, s * s, X_bar, U)
            Z = P_U @ Z
            grad_A = (
                grad_A
                + Z @ s_diag @ Vh
                + U @ Z.mH @ A @ P_V
                + U @ s_inv_diag @ P_V_gV.mH
            )
        else:
            Y_bar = P_V @ (P_V_gV + A.mH @ P_U_gU @ s_inv_diag)
            W = _solve_sylvester_colwise(A.mH @ A, s * s, Y_bar, V)
            W = P_V @ W
            grad_A = (
                grad_A + U @ s_diag @ W.mH + P_U @ A @ W @ Vh + P_U_gU @ s_inv_diag @ Vh
            )
    elif k == min(n, m):
        if n > m:
            P_U = torch.eye(n, dtype=A.dtype, device=A.device) - U @ U.mH
            grad_A = grad_A + P_U @ grad_U @ s_inv_diag @ Vh
        elif n < m:
            P_V = torch.eye(m, dtype=A.dtype, device=A.device) - V @ V.mH
            grad_A = grad_A + U @ s_inv_diag @ (P_V @ grad_V).mH

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
