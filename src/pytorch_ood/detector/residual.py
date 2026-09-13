"""

.. image:: https://img.shields.io/badge/classification-yes-brightgreen?style=flat-square
   :alt: classification badge
.. image:: https://img.shields.io/badge/segmentation-no-red?style=flat-square
   :alt: classification badge

..  autoclass:: pytorch_ood.detector.Residual
    :members:
    :inherited-members:
    :show-inheritance:

"""

import logging
from typing import Callable, Optional, TypeVar

import numpy as np
import torch
import torch.nn as nn
from numpy.linalg import norm, pinv
from sklearn.covariance import EmpiricalCovariance
from torch import Tensor
from torch.utils.data import DataLoader

from ..api import Detector, ModelNotSetException, RequiresFittingException
from ..utils import extract_features

log = logging.getLogger(__name__)
Self = TypeVar("Self")


class Residual(Detector):
    """
    Implements the Residual detector from the paper
    *ViM: Out-Of-Distribution with Virtual-logit Matching* (Wang et al., CVPR 2022).

    Projects penultimate-layer features onto the **null subspace** — the subspace
    spanned by the eigenvectors of the empirical covariance matrix that correspond
    to the **smallest** eigenvalues (i.e. low-variance directions not explained by
    the training data). The L2 norm of those projected features serves as the OOD score.

    Specifically, given the weight matrix :math:`W` and bias :math:`b` of the final
    linear layer, a new origin is computed as :math:`u = -W^+b`, where :math:`W^+`
    denotes the Moore-Penrose pseudo-inverse. The residual null subspace :math:`NS`
    is the matrix whose columns are the eigenvectors of the empirical covariance of
    :math:`(z - u)` that correspond to the :math:`D - d` smallest eigenvalues.

    The outlier score for a sample with features :math:`z` is:

    .. math ::
        s(x) = ||(z - u) @ NS||_2

    Higher scores indicate more likely OOD samples.

    :see Paper: `CVPR 2022 <https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_ViM_Out-Of-Distribution_With_Virtual-Logit_Matching_CVPR_2022_paper.pdf>`__
    :see Implementation: `GitHub <https://github.com/haoqiwang/vim>`__
    """

    requires_fit = True

    def __init__(
        self,
        net: nn.Module,
        d: int = 128,
    ):
        """
        :param net: the neural network. Must expose a ``fc`` attribute (final linear layer)
            with ``weight`` and ``bias`` tensors, as is standard for ResNet models.
        :param d: dimensionality of the principal subspace to discard. The null subspace
            will have dimension ``D - d`` where ``D`` is the feature dimensionality.
        """
        super().__init__()
        self.net = net
        self.d = d

        w = net.fc.weight.data.cpu().numpy()  # (C, D)
        b = net.fc.bias.data.cpu().numpy()    # (C,)
        self.u: np.ndarray = -np.matmul(pinv(w), b)   # (D,)  new origin

        self._feature_extractor = torch.nn.Sequential(
            *list(net.children())[:-1], torch.nn.Flatten()
        )
        self._NS: Optional[np.ndarray] = None   # null subspace matrix (D, D-d)

    def fit(self: Self, data_loader: DataLoader) -> Self:
        """
        Compute the empirical covariance of centered training features and derive the
        null subspace. Ignores any OOD-labelled samples in the loader.

        :param data_loader: DataLoader with in-distribution training data.
        """
        device = next(self.net.parameters()).device
        z, _ = extract_features(data_loader, self._feature_extractor, device=str(device))
        features = z.cpu().numpy()

        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(features - self.u)

        eig_vals, eig_vecs = np.linalg.eig(ec.covariance_)

        # Sort eigenvalues descending; the null subspace is the last (D-d) eigenvectors
        sorted_idx = np.argsort(eig_vals * -1)          # descending
        null_vecs = eig_vecs.T[sorted_idx[self.d:]]     # (D-d, D)
        self._NS = np.ascontiguousarray(null_vecs.T)     # (D, D-d)

        return self

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        """
        Compute the Residual OOD score for a batch of inputs.

        :param x: input tensor (batch of images).
        :return: outlier scores — higher values indicate more likely OOD.
        """
        if self._NS is None:
            raise RequiresFittingException()

        features = self._feature_extractor(x).cpu().numpy()   # (N, D)
        score = norm(np.matmul(features - self.u, self._NS), axis=-1)  # (N,)
        return torch.from_numpy(score).float()
