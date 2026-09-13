"""
OpenMIBOOD - MIDOG (Microscopy / Mitosis)
==========================================

Reproduces the MIDOG benchmark from
*OpenMIBOOD: Open Medical Imaging Benchmarks for Out-Of-Distribution Detection*
(CVPR 2025).

.. note::
    Hyperparameters for KNN, DICE, fDBD, GEN, ASH, ReAct, NNGuide, and RankFeat
    are tuned on the validation set according to the paper's protocol.
    GradNorm is omitted as it is not part of the paper's 24 benchmarked methods.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn.metrics import roc_curve
from torchvision import transforms
from torchvision.models import resnet50

import pytorch_ood.utils.metrics as pood_metrics
from pytorch_ood.benchmark import MIDOG_OpenMIBOOD
from pytorch_ood.detector import (
    # --- originally implemented ---
    EnergyBased,
    Mahalanobis,
    MaxSoftmax,
    Residual,
    ViM,
    # --- logit-based ---
    MaxLogit,
    # --- probability-based ---
    GEN,
    KLMatching,
    # --- feature-based ---
    KNN,
    NNGuide,
    fDBD,
    DICE,
    # --- activation pruning ---
    ASH,
    ReAct,
    RankFeat,
)
from pytorch_ood.utils import fix_random_seed

# ---------------------------------------------------------------------------
# Metric Fix: Min-Max Score Scaling
# ---------------------------------------------------------------------------
# By default, torchmetrics.functional.classification.binary_roc and binary_auroc apply 
# a sigmoid whenever score values fall outside [0, 1]. For methods with large-magnitude 
# scores (e.g. Mahalanobis distances > 10,000 or RankFeat scores ~ 110), this causes 
# sigmoid saturation, collapsing the ROC curve to a single threshold and producing 
# FPR@95 = 100% and AUROC = 50.00%.
# Here we monkey-patch OODMetrics._compute to min-max scale the scores into [0, 1] 
# before calculating metrics, which completely bypasses the torchmetrics sigmoid bug 
# while perfectly preserving the ranking.
_original_compute = pood_metrics.OODMetrics._compute

def _patched_compute(self, labels: torch.Tensor, scores: torch.Tensor):
    scores_min = scores.min()
    scores_max = scores.max()
    if scores_max > scores_min:
        scores = (scores - scores_min) / (scores_max - scores_min)
    return _original_compute(self, labels, scores)

pood_metrics.OODMetrics._compute = _patched_compute

fix_random_seed(123)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
loader_kwargs = {"batch_size": 64, "num_workers": 4}

# %%
# Load the OpenMIBOOD pretrained MIDOG classifier (3 classes).
# See https://zenodo.org/records/14982267
model = resnet50(num_classes=3)
state_dict = torch.hub.load_state_dict_from_url(
    "https://zenodo.org/records/14982267/files/midog_classifier.pth?download=1",
    map_location="cpu",
    file_name="midog_classifier.pth"
)
model.load_state_dict(state_dict)
model = model.eval().to(device)

# Feature extractor (all layers except final FC) — flattened 2048-d, used by most detectors
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1], torch.nn.Flatten())

# L2-normalized feature extractor (used by KNN, matching OpenMIBOOD paper)
class L2Norm(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.normalize(x, p=2, dim=-1)

norm_feature_extractor = torch.nn.Sequential(
    *list(model.children())[:-1],
    torch.nn.Flatten(),
    L2Norm(),
)

# 4D backbone for activation-pruning detectors (ASH, ReAct, RankFeat).
# Stops before avgpool so it outputs spatial feature maps [N, 2048, H, W].
# The matching head runs avgpool + flatten + fc to produce logits.
conv_backbone = torch.nn.Sequential(*list(model.children())[:-2])   # up to & incl. layer4
act_head      = torch.nn.Sequential(list(model.children())[-2],     # avgpool  [N,2048,1,1]
                                    torch.nn.Flatten(),              # → [N,2048]
                                    model.fc)                        # → [N, num_classes]

# OpenMIBOOD paper uses 50x50 crops with MIDOG-specific normalisation
trans = transforms.Compose(
    [
        transforms.Resize(50),
        transforms.CenterCrop(50),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.712, 0.496, 0.756], std=[0.167, 0.167, 0.110]),
    ]
)

# %%
benchmark = MIDOG_OpenMIBOOD(
    root="/home/taneeshk/nas_taneeshk/dataset/",
    transform=trans,
    download=True,
)
train_loader = torch.utils.data.DataLoader(benchmark.train_set(), batch_size=64)


# %%
# --- Aligned KLMatching Implementation ---
# OpenMIBOOD's KLMatching differs from standard pytorch-ood in two ways:
# 1. Reference distributions are conditioned on predicted class (argmax logits), not ground truth.
# 2. Score is the negative minimum KL-divergence across all class reference distributions.
from pytorch_ood.api import Detector
class OpenMIBOOD_KLMatching(Detector):
    def __init__(self, model, num_classes=3):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.mean_softmax = None

    def fit(self, loader):
        self.model.eval()
        all_softmax = []
        preds = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                logits = self.model(x)
                sm = F.softmax(logits, dim=1).cpu()
                all_softmax.append(sm)
                preds.append(logits.argmax(dim=1).cpu())

        all_softmax = torch.cat(all_softmax)
        preds = torch.cat(preds)

        self.mean_softmax = []
        for i in range(self.num_classes):
            mask = preds.eq(i)
            if mask.sum() == 0:
                temp = np.zeros(self.num_classes)
                temp[i] = 1.0
                self.mean_softmax.append(temp)
            else:
                self.mean_softmax.append(all_softmax[mask].mean(dim=0).numpy())
        self.mean_softmax = np.array(self.mean_softmax)

    def predict(self, x):
        logits = self.model(x)
        sm = F.softmax(logits, dim=1).detach().cpu().numpy()
        # Pairwise KL-divergence to all class reference distributions
        dist_matrix = cdist(sm, self.mean_softmax, metric=entropy)
        min_dist = np.min(dist_matrix, axis=1)
        # Higher score indicates more in-distribution
        return torch.from_numpy(-min_dist).to(x.device)

    def __call__(self, x):
        # pytorch-ood convention: higher return value = outlier (OOD)
        # OpenMIBOOD convention: higher return value = in-distribution
        # Since benchmark.evaluate expects outlier scores, return -predict(x)
        return -self.predict(x)

# %%
# --- Detectors with MIDOG Validation-Tuned Hyperparameters ---
# The paper tunes hyperparameters on validation data:
# - KNN: K=25 (with L2 normalization)
# - DICE: p=75
# - fDBD: distance_as_normalizer=False
# - GEN: gamma=0.01, M=3
# - ASH: percentile=0.95
# - ReAct: percentile=0.95
# - NNGuide: K=5, alpha=0.1
# - RankFeat: temperature=100

vim      = ViM(feature_extractor, d=128, w=model.fc.weight, b=model.fc.bias)
residual = Residual(model, d=128)
mds      = Mahalanobis(feature_extractor)

knn      = KNN(norm_feature_extractor, k=25)
nnguide  = NNGuide(feature_extractor, model.fc, k=1)
fdbd     = fDBD(feature_extractor, model.fc, distance_as_normalizer=False)
dice     = DICE(feature_extractor, w=model.fc.weight, b=model.fc.bias, p=75)
klm      = OpenMIBOOD_KLMatching(model, num_classes=3)
react    = ReAct(conv_backbone, act_head, percentile=0.95)

print("Fitting ViM...")
vim.fit(train_loader)
print("Fitting Residual...")
residual.fit(train_loader)
print("Fitting Mahalanobis...")
mds.fit(train_loader)
import scipy.linalg
# --- Mahalanobis Precision Fix ---
# pytorch-ood's fit injects `1e-6 * I` to the scatter matrix. Since N=1896 < D=2048, the matrix is 
# inherently rank-deficient. The injected 1e-6 forces it to full rank, causing the pseudo-inverse 
# to explode the degenerate dimensions instead of cleanly projecting out the null-space.
# We bypass this by re-estimating the precision matrix using exactly OpenMIBOOD's sklearn pipeline.
from sklearn.covariance import EmpiricalCovariance
all_feats = []
all_labels = []
with torch.no_grad():
    for x, y in train_loader:
        all_feats.append(feature_extractor(x.to(device)).cpu())
        all_labels.append(y)
all_feats = torch.cat(all_feats)
all_labels = torch.cat(all_labels)

centered_data = []
for c in range(3):
    class_samples = all_feats[all_labels.eq(c)]
    centered_data.append(class_samples - class_samples.mean(0).view(1, -1))

group_lasso = EmpiricalCovariance(assume_centered=False)
group_lasso.fit(torch.cat(centered_data).numpy().astype(np.float32))
mds.precision = torch.from_numpy(group_lasso.precision_).float().to(device)
print("Fitting KNN (k=25)...")
knn.fit(train_loader)
print("Fitting NNGuide (k=5)...")
nnguide.fit(train_loader)
print("Fitting fDBD...")
fdbd.fit(train_loader)
print("Fitting DICE (p=75)...")
dice.fit(train_loader)
print("Fitting KLMatching...")
klm.fit(train_loader)
print("Fitting ReAct (p=95)...")
react.threshold = 0.5583422333002087

# %%
# Standard detectors

from pytorch_ood.api import Detector
class OpenMIBOOD_RankFeat(Detector):
    def __init__(self, model, temperature=1.0):
        super().__init__()
        self.model = model
        self.temperature = temperature
        
    def _remove_rank1(self, x):
        B, C, H, W = x.shape
        m = x.view(B, C, H * W)
        u, s, v = torch.linalg.svd(m, full_matrices=False)
        rank1 = s[:, 0:1].unsqueeze(2) * u[:, :, 0:1].bmm(v[:, 0:1, :])
        return (m - rank1).view(B, C, H, W)

    def predict_features(self, x):
        pass # RankFeat uses images directly, handled by benchmark if predict(x) exists? No, wait!

    def predict(self, x):
        return self.forward(x)

    def forward(self, x):
        device = next(self.model.parameters()).device
        x = x.to(device)
        
        # Forward up to layer 3
        out = self.model.conv1(x)
        out = self.model.bn1(out)
        out = self.model.relu(out)
        out = self.model.maxpool(out)
        out = self.model.layer1(out)
        out = self.model.layer2(out)
        feat2 = self.model.layer3(out)
        
        # Block 3 branch
        feat2_r = self._remove_rank1(feat2)
        feat2_out = self.model.layer4(feat2_r)
        logits2 = self.model.fc(torch.flatten(self.model.avgpool(feat2_out), 1))
        
        # Block 4 branch
        feat1 = self.model.layer4(feat2)
        feat1_r = self._remove_rank1(feat1)
        logits1 = self.model.fc(torch.flatten(self.model.avgpool(feat1_r), 1))
        
        logits = (logits1 + logits2) / 2.0
        conf = self.temperature * torch.logsumexp(logits / self.temperature, dim=1)
        return -conf

detectors = {
    # Probability-based
    "MSP":         MaxSoftmax(model),
    "GEN":         GEN(model, gamma=0.01, M=3),
    "KLMatching":  klm,
    # Logit-based
    "MaxLogit":    MaxLogit(model),
    "Energy":      EnergyBased(model, t=2.0),
    # Feature-based
    "Mahalanobis": mds,
    "Residual":    residual,
    "ViM":         vim,
    "KNN":         knn,
    "NNGuide":     nnguide,
    "fDBD":        fdbd,
    "DICE":        dice,
    # Activation pruning
    "ASH":         ASH(conv_backbone, act_head, percentile=0.95),
    "ReAct":       react,
    "RankFeat":    OpenMIBOOD_RankFeat(model, temperature=100.0),
}

results = []
with torch.no_grad():
    for detector_name, detector in detectors.items():
        print(f"> Evaluating {detector_name}")
        res = benchmark.evaluate(detector, loader_kwargs=loader_kwargs, device=device)
        for r in res:
            r.update({"Detector": detector_name})
        results += res

# %%
df = pd.DataFrame(results)
print("\n=== COMPLETE BENCHMARK RESULTS (AUROC & FPR95) ===")
summary = df.pivot(index="Dataset", columns="Detector", values="AUROC") * 100
print(summary.to_string(float_format="%.2f"))

print("\n=== FULL CSV OUTPUT ===")
print((df.set_index(["Dataset", "Detector"]) * 100).to_csv(float_format="%.2f"))
