import torch
from typing import List, Tuple, Dict, Sequence, Optional
from rtd import RTD


class RTDAttraction(torch.nn.Module):
    """
    A wrapper that receives a list of (tensor, tensor) pairs and returns the
    summed RTD distances (optionally weighted).
    """
    def __init__(self,
                 pairs: Sequence[Tuple[str, str]],
                 layer_map: Dict[str, torch.nn.Module],
                 coeff: float = 1.,
                 maxdim: int = 1,
                 q: float = 1.,
                 stop_before: bool = False,
                 stop_after: bool = False):
        super().__init__()
        self.pairs = pairs
        self.coeff = coeff
        self.topo = RTD(maxdim=maxdim, q=q)
        self.layer_map = layer_map
        self.stop_before = stop_before
        self.stop_after = stop_after
        self._register_hooks()

    def _register_hooks(self):
        self.act: Dict[str, torch.Tensor] = {}

        def save(name):
            def _hook(_, __, output):
                self.act[name] = output.detach() if self.stop_after else output
            return _hook

        for name, layer in self.layer_map.items():
            layer.register_forward_hook(save(name))

    def forward(self) -> torch.Tensor:
        loss = 0.0
        for a, b in self.pairs:
            Xa = self.act[a].reshape(self.act[a].shape[0], -1)
            Xb = self.act[b].detach() if self.stop_before else self.act[b]
            Xb = Xb.reshape(Xb.shape[0], -1)
            loss = loss + self.topo(Xa, Xb).sum()
        return self.coeff * loss


def build_regulariser(config_id: int,
                      model: torch.nn.Module,
                      coeff: float,
                      grid: int = 1,
                      stop_grad: str | None = None) -> Optional[torch.nn.Module]:
    """
    Returns None for configurations 1 & 2.
    For 3,4,5 returns an RTDAttraction instance with the required pairs.
    """
    layers = {name: layer for name, layer in model.named_modules()
              if isinstance(layer, (torch.nn.Conv2d,
                                    torch.nn.ReLU,
                                    torch.nn.BatchNorm2d,
                                    torch.nn.MaxPool2d,
                                    torch.nn.Linear))}

    names = list(layers.keys())

    if config_id == 3:
        pairs = [(names[0], names[-2])]           # input ↔ last hidden
    elif config_id == 4:
        pairs = [(n, names[0]) for n in names] + \
                [(n, names[-2]) for n in names]   # each to both ends
    elif config_id == 5:
        pairs = []
        for i in range(0, len(names) - grid, grid):
            pairs.append((names[i], names[i + grid]))
    else:
        return None

    return RTDAttraction(
        pairs=pairs,
        layer_map=layers,
        coeff=coeff,
        stop_before=stop_grad == "before",
        stop_after=stop_grad == "after"
    )
