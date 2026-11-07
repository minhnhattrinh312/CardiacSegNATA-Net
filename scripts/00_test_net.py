import torch.nn as nn
import torch
from pathlib import Path
import sys

src_dir = Path(__file__).resolve().parents[1] / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
# location of fvcore (not needed if it is installed in your env)
from fvcore.nn import FlopCountAnalysis # type: ignore

import importlib.util
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Fixed(nn.Module):
    """forward(x, view) takes a string, so fix the view for FLOP counting."""

    def __init__(self, net, view):
        super().__init__()
        self.net, self.view = net, view

    def forward(self, x):
        return self.net(x, self.view)


root = "/home/nhattm/CardiacSegNATA-Net/src/segment2d/"
eff = load(root + "EfficientNATA_Net.py", "eff")
base = load(root + "nata_net.py", "base")

x = torch.randn(1, 1, 256, 256, requires_grad=False).cuda()
for name, net in [("EfficientNATA_Net", eff.EfficientNATA_Net()), ("NATA_Net (baseline)", base.NATA_Net())]:
    net = net.cuda().eval()
    print(f"== {name}")
    print("params:", count_parameters(net))
    with torch.no_grad():
        for view in net.finalHeads:
            print(f"  {view}: out {tuple(net(x, view).shape)}")
        flops = FlopCountAnalysis(Fixed(net, "SAX"), x)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        print("flops (SAX head):", flops.total(), f"= {flops.total()/1e9:.2f} G")
