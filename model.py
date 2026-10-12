import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from config import DEVICE, INPUT_DIM, N_CLASSES


class ActivityMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, N_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


def new_model(state=None):
    model = ActivityMLP().to(DEVICE)
    if state is not None:
        model.load_state_dict(state)
    return model


def state_to_cpu(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def evaluate(model, dataset):
    model.eval()
    loader = DataLoader(dataset, batch_size=2048, shuffle=False)
    logits, labels = [], []

    for x, y in loader:
        logits.append(model(x.to(DEVICE)).cpu())
        labels.append(y)

    logits = torch.cat(logits)
    labels = torch.cat(labels)
    pred = logits.argmax(dim=1).numpy()
    y = labels.numpy()

    return {
        "loss": nn.CrossEntropyLoss()(logits, labels).item(),
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
    }
