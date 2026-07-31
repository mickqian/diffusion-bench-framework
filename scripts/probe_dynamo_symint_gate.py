"""Does the gate survive when its dim comes from a module attribute (what the
real call sites would use) rather than a compiled-function argument?"""
import torch

log = []


@torch.compiler.assume_constant_result
def gate(dim: int) -> bool:
    log.append(("gate", dim, type(dim).__name__))
    return dim % 2 == 0


class FromAttr(torch.nn.Module):
    def __init__(self, rope_dim):
        super().__init__()
        self.rope_dim = rope_dim

    def forward(self, x, cache):
        ok = gate(self.rope_dim)          # module attribute -> should be constant
        return x + (1 if ok else 0)


class FromTensor(torch.nn.Module):
    def forward(self, x, cache):
        ok = gate(cache.size(-1))        # today's code
        return x + (1 if ok else 0)


for name, mod in (("attr", FromAttr(64)), ("tensor", FromTensor())):
    for dyn in (True, None):
        log.clear()
        c = torch.compile(mod, dynamic=dyn)
        try:
            c(torch.zeros(4), torch.zeros(8, 64))
            c(torch.zeros(4), torch.zeros(16, 64))   # seq dim varies, width fixed
            c(torch.zeros(4), torch.zeros(32, 128))  # width varies too
            print(f"{name:7s} dynamic={str(dyn):5s} OK    {log}")
        except Exception as e:
            print(f"{name:7s} dynamic={str(dyn):5s} RAISE {type(e).__name__}: {str(e)[:60]}")
