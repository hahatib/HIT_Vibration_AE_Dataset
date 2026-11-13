import math
import torch
from torch import nn
from torch.nn import functional as F


def branch_stats(p):
    top = p.topk(2, dim=1).values
    entropy = -(p * p.clamp_min(1e-12).log()).sum(1, keepdim=True) / math.log(4)
    return torch.cat((entropy, top[:, :1] - top[:, 1:2]), dim=1)


class Calibrator(nn.Module):

    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(13, hidden), nn.Tanh(), nn.Linear(hidden, 5))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        lo, hi = (math.log(0.2), math.log(8.0))
        initial = -lo / (hi - lo)
        self.offset = math.log(initial / (1 - initial))

    def forward(self, p, context):
        logp = p.clamp_min(1e-12).log()
        centered = (logp - logp.mean(1, keepdim=True)) / 8.0
        x = torch.cat((p, centered, branch_stats(p), context), dim=1)
        z = self.net(x)
        logt = math.log(0.2) + math.log(40.0) * torch.sigmoid(z[:, :1] + self.offset)
        bias = 0.75 * torch.tanh(z[:, 1:])
        bias = bias - bias.mean(1, keepdim=True)
        logq = F.log_softmax(logp / logt.exp() + bias, dim=1)
        return (logq, logt, bias)


class SerialFusion(nn.Module):

    def __init__(self, method, hidden=8):
        super().__init__()
        assert method in ("Full", "Calibration only", "Reliability only")
        self.method = method
        if method != "Reliability only":
            self.cal_a = Calibrator(hidden)
            self.cal_b = Calibrator(hidden)
        if method != "Calibration only":
            self.reliability = nn.Sequential(
                nn.Linear(17, hidden), nn.Tanh(), nn.Linear(hidden, 2)
            )
            nn.init.zeros_(self.reliability[-1].weight)
            nn.init.constant_(self.reliability[-1].bias, math.log(0.97 / 0.03))

    def forward(self, a, b, context):
        if self.method == "Reliability only":
            loga, logb = (a.clamp_min(1e-12).log(), b.clamp_min(1e-12).log())
            loga, logb = (F.log_softmax(loga, 1), F.log_softmax(logb, 1))
            logt = torch.zeros((len(a), 2), device=a.device)
            bias = torch.zeros((len(a), 8), device=a.device)
        else:
            loga, ta, ba = self.cal_a(a, context)
            logb, tb, bb = self.cal_b(b, context)
            logt, bias = (torch.cat((ta, tb), 1), torch.cat((ba, bb), 1))
        qa, qb = (loga.exp(), logb.exp())
        if self.method == "Calibration only":
            reliability = torch.ones((len(a), 2), device=a.device)
            logfusion = F.log_softmax(loga + logb, dim=1)
        else:
            agreement = (qa.argmax(1) == qb.argmax(1)).float()[:, None]
            x = torch.cat(
                (
                    qa,
                    qb,
                    branch_stats(qa),
                    branch_stats(qb),
                    (qa * qb).sum(1, keepdim=True),
                    agreement,
                    context,
                ),
                dim=1,
            )
            reliability = 0.01 + 0.9899 * torch.sigmoid(self.reliability(x))
            ra, rb = (reliability[:, :1], reliability[:, 1:2])
            singleton = ra * rb * qa * qb + ra * (1 - rb) * qa + (1 - ra) * rb * qb
            numerator = singleton + (1 - ra) * (1 - rb) / 4.0
            probability = numerator / numerator.sum(1, keepdim=True)
            logfusion = probability.log()
        return {
            "logp": logfusion,
            "loga": loga,
            "logb": logb,
            "r": reliability,
            "logt": logt,
            "bias": bias,
        }


def objective(model, output, y, weights):
    loss = F.nll_loss(output["logp"], y, reduction="none")
    if model.method != "Reliability only":
        branch = (
            F.nll_loss(output["loga"], y, reduction="none")
            + F.nll_loss(output["logb"], y, reduction="none")
        ) / 2
        loss = loss + 0.1 * branch + 0.01 * output["logt"].square().mean(1)
        loss = loss + 0.05 * output["bias"].square().mean(1)
    if model.method != "Calibration only":
        target = (
            torch.stack(
                (output["loga"].argmax(1) == y, output["logb"].argmax(1) == y), dim=1
            )
            .float()
            .detach()
        )
        loss = loss + 0.05 * F.binary_cross_entropy(
            output["r"], target, reduction="none"
        ).mean(1)
    return (weights * loss).mean()
