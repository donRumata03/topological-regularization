import torch
from torch_topological.nn import VietorisRipsComplex


def make_r_cross_matrix(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Build the (2n+1)×(2n+1) 'relative-cross' distance matrix.
    """
    n = X.shape[0]
    XX = torch.cdist(X, X)  # n×n
    YY = torch.cdist(Y, Y)  # n×n

    inf = float('inf')
    inf_block = torch.triu(torch.full_like(XX, inf), diagonal=1) + XX

    zero_col = torch.zeros(n, 1, device=X.device, dtype=X.dtype)
    zero_row = torch.zeros(1, n, device=X.device, dtype=X.dtype)
    inf_col = torch.full((n, 1), inf, device=X.device, dtype=X.dtype)
    inf_row = torch.full((1, n), inf, device=X.device, dtype=X.dtype)

    upper = torch.cat((XX, inf_block.t(), zero_col), dim=1)
    mid = torch.cat((inf_block, torch.minimum(XX, YY), inf_col), dim=1)
    lower = torch.cat((zero_row, inf_row, torch.zeros(1, 1, device=X.device, dtype=X.dtype)), dim=1)
    M = torch.cat((upper, mid, lower), dim=0)

    big = (M[torch.isfinite(M)].max() + 1.0).item()
    M = torch.where(torch.isfinite(M), M, torch.full_like(M, big))
    return M


class RTD(torch.nn.Module):
    """
    Representation Topology Distance layer implemented with pytorch-topological.
    Returns one value per homology dimension in [0, maxdim].
    """

    def __init__(self, maxdim: int = 1, q: float = 1.0):
        super().__init__()
        self.maxdim = maxdim
        self.q = q
        self.vr = VietorisRipsComplex(dim=maxdim)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        M = make_r_cross_matrix(X, Y)  # (2n+1)×(2n+1)

        # Batch dimension of size 1 and flag ‘treat_as_distances’
        batch_pi = self.vr(M.unsqueeze(0), treat_as_distances=True)

        # Unwrap batch dimension
        diagrams = batch_pi[0]  # list with len == maxdim+1

        tot_pers = []
        for dim in range(self.maxdim + 1):
            # convert births & deaths to torch tensor
            diag_np = diagrams[dim].diagram  # NumPy array
            diag = torch.as_tensor(diag_np,
                                   device=X.device,
                                   dtype=X.dtype)

            if diag.numel() == 0:  # empty diagram
                tot_pers.append(torch.zeros((), device=X.device, dtype=X.dtype))
            else:
                life = torch.clamp(diag[:, 1] - diag[:, 0], min=0.)
                tot_pers.append((life.pow(self.q)).sum())

        return torch.stack(tot_pers)


# ------------------------------------------------------------------ demo
if __name__ == "__main__":
    torch.manual_seed(0)
    n, d = 16, 3
    X = torch.randn(n, d, requires_grad=True)
    Y = torch.randn(n, d, requires_grad=True)

    rtd_layer = RTD(maxdim=1, q=1.0)
    score = rtd_layer(X, Y).sum()  # scalar score
    score.backward()

    print("RTD value :", score.item())
    print("‖∂/∂X‖₂   :", X.grad.norm().item())
    print("‖∂/∂Y‖₂   :", Y.grad.norm().item())
    print(rtd_layer(X, X.exp() * 1000000000.1 + X).sum())
