from abc import ABC, abstractmethod
from typing import NamedTuple

from numpy import ndarray
from ..tools import check_perm_blocks_dim, chi2_approx, compute_dist
from ._utils import _CheckInputs


class KSampleTestOutput(NamedTuple):
    stat: float
    pvalue: float


class KSampleTest(ABC):
    """
    A base class for a *k*-sample test.

    Parameters
    ----------
    compute_distance : str, callable, or None, default: "euclidean" or "gaussian"
        A function that computes the distance among the samples within each
        data matrix.
        Valid strings for ``compute_distance`` are, as defined in
        :func:`sklearn.metrics.pairwise_distances`,

            - From scikit-learn: [``"euclidean"``, ``"cityblock"``, ``"cosine"``,
              ``"l1"``, ``"l2"``, ``"manhattan"``] See the documentation for
              :mod:`scipy.spatial.distance` for details
              on these metrics.
            - From scipy.spatial.distance: [``"braycurtis"``, ``"canberra"``,
              ``"chebyshev"``, ``"correlation"``, ``"dice"``, ``"hamming"``,
              ``"jaccard"``, ``"kulsinski"``, ``"mahalanobis"``, ``"minkowski"``,
              ``"rogerstanimoto"``, ``"russellrao"``, ``"seuclidean"``,
              ``"sokalmichener"``, ``"sokalsneath"``, ``"sqeuclidean"``,
              ``"yule"``] See the documentation for :mod:`scipy.spatial.distance` for
              details on these metrics.

        Alternatively, this function computes the kernel similarity among the
        samples within each data matrix.
        Valid strings for ``compute_kernel`` are, as defined in
        :func:`sklearn.metrics.pairwise.pairwise_kernels`,

            [``"additive_chi2"``, ``"chi2"``, ``"linear"``, ``"poly"``,
            ``"polynomial"``, ``"rbf"``,
            ``"laplacian"``, ``"sigmoid"``, ``"cosine"``]

        Note ``"rbf"`` and ``"gaussian"`` are the same metric.
    bias : bool (default: False)
        Whether or not to use the biased or unbiased test statistics. Only
        applies to ``Dcorr`` and ``Hsic``.
    **kwargs
        Arbitrary keyword arguments for ``compute_distkern``.
    """

    def __init__(self, compute_distance=None, bias=False, **kwargs):
        # set statistic and p-value
        self.stat = None
        self.pvalue = None
        self.bias = bias
        self.compute_distance = compute_distance
        self.kwargs = kwargs

        super().__init__()

    @abstractmethod
    def statistic(self, *args):
        r"""
        Calulates the *k*-sample test statistic.

        Parameters
        ----------
        *args : ndarray of float
            Variable length input data matrices. All inputs must have the same
            number of dimensions. That is, the shapes must be `(n, p)` and
            `(m, p)`, ... where `n`, `m`, ... are the number of samples and `p` is
            the number of dimensions.

        Returns
        -------
        stat : float
            The computed *k*-Sample statistic.
        """

    @abstractmethod
    def test(self, *args, reps=1000, workers=1, random_state=None, perm_blocks=None, auto=True):
        r"""
        Calculates the *k*-sample test statistic and p-value.

        Parameters
        ----------
        *args : ndarray of float
            Variable length input data matrices. All inputs must have the same
            number of dimensions. That is, the shapes must be `(n, p)` and
            `(m, p)`, ... where `n`, `m`, ... are the number of samples and `p` is
            the number of dimensions.
        reps : int, default: 1000
            The number of replications used to estimate the null distribution
            when using the permutation test used to calculate the p-value.
        workers : int, default: 1
            The number of cores to parallelize the p-value computation over.
            Supply ``-1`` to use all cores available to the Process.

        Returns
        -------
        stat : float
            The computed *k*-sample statistic.
        pvalue : float
            The computed *k*-sample p-value.
        """
        check_input = _CheckInputs(
            x,
            y,
            reps=reps,
        )
        x, y = check_input()
        if perm_blocks is not None:
            check_perm_blocks_dim(perm_blocks, y)

        if (
            auto
            and x.shape[1] == 1
            and y.shape[1] == 1
            and self.compute_distance == "euclidean"
        ):
            self.is_fast = True

        if auto and x.shape[0] > 20 and perm_blocks is None:
            stat, pvalue = chi2_approx(self.statistic, x, y)
            self.stat = stat
            self.pvalue = pvalue
            self.null_dist = None
        else:
            if not self.is_fast:
                x, y = compute_dist(
                    x, y, metric=self.compute_distance, **self.kwargs)
                self.is_distance = True
            stat, pvalue = super(KSampleTest, self).test(
                x,
                y,
                reps,
                workers,
                perm_blocks=perm_blocks,
                is_distsim=self.is_distance,
                random_state=random_state,
            )

        return KSampleTestOutput(stat, pvalue)
