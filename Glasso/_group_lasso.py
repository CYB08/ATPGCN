import warnings
from abc import ABC, abstractmethod
from math import sqrt
from numbers import Number

import numpy as np
import numpy.linalg as la
from scipy import sparse
from scipy.special import logsumexp
from sklearn.base import (BaseEstimator, ClassifierMixin, RegressorMixin,
                          TransformerMixin)
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import (check_array, check_consistent_length,
                           check_random_state)
from sklearn.utils.multiclass import unique_labels
from sklearn.exceptions import NotFittedError

from Glasso._fista import FISTAProblem
from Glasso._singular_values import find_largest_singular_value
from Glasso._subsampling import Subsampler

_DEBUG = False

def _l1_l2_prox(w, l1_reg, group_reg, groups):
    return _group_l2_prox(_l1_prox(w, l1_reg), group_reg, groups)


def _l1_prox(w, reg):
    return np.sign(w) * np.maximum(0, np.abs(w) - reg)


def _l2_prox(w, reg):
    """The proximal operator for reg*||w||_2 (not squared).
    """
    norm_w = la.norm(w)
    if norm_w == 0:
        return 0 * w
    return max(0, 1 - reg / norm_w) * w


def _group_l2_prox(w, reg_coeffs, groups):
    """The proximal map for the specified groups of coefficients.
    """
    w = w.copy()

    for group, reg in zip(groups, reg_coeffs):
        w[group] = _l2_prox(w[group], reg)

    return w


def _split_intercept(w):
    return w[0], w[1:]


def _join_intercept(b, w):
    num_classes = w.shape[1]
    return np.concatenate([np.array(b).reshape(1, num_classes), w], axis=0)


def _add_intercept_col(X):
    ones = np.ones([X.shape[0], 1])
    if sparse.issparse(X):
        return sparse.hstack((ones, X))
    return np.hstack([ones, X])


def _parse_group_iterable(iterable_or_number):
	try:
		iter(iterable_or_number)
	except TypeError:
		if iterable_or_number is None:
			return -1
		else:
			return iterable_or_number
	else:
		return [_parse_group_iterable(i) for i in iterable_or_number]


class BaseGroupLasso(ABC, BaseEstimator, TransformerMixin):

    LOG_LOSSES = False

    def __init__(
        self,
        groups=None,
        group_reg=0.05,
        l1_reg=0.00,
        n_iter=100,
        tol=1e-5,
        scale_reg="group_size",
        subsampling_scheme=None,
        fit_intercept=True,
        random_state=None,
        warm_start=False,
        old_regularisation=False,
        supress_warning=False,
    ):
        self.groups = groups
        self.group_reg = group_reg
        self.scale_reg = scale_reg
        self.l1_reg = l1_reg
        self.n_iter = n_iter
        self.tol = tol
        self.subsampling_scheme = subsampling_scheme
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.old_regularisation = old_regularisation
        self.warm_start = warm_start
        self.supress_warning = supress_warning

    def _more_tags(self):
        return {'multioutput': True}

    def _regulariser(self, w):
        """The regularisation penalty for a given coefficient vector, ``w``.

        The first element of the coefficient vector is the intercept which
        is sliced away.
        """
        regulariser = 0
        coef_ = _split_intercept(w)[1]
        for group, reg in zip(self.groups_, self.group_reg_vector_):
            regulariser += reg * la.norm(coef_[group])
        regulariser += self.l1_reg * la.norm(coef_.ravel(), 1)
        return regulariser

    def _get_reg_strength(self, group, reg):
        """Get the regularisation coefficient for one group.
        """
        scale_reg = str(self.scale_reg).lower()
        if scale_reg == "group_size":
            scale = sqrt(group.sum())
        elif scale_reg == "none":
            scale = 1
        elif scale_reg == "inverse_group_size":
            scale = 1 / sqrt(group.sum())
        else:
            raise ValueError(
                '``scale_reg`` must be equal to "group_size",'
                ' "inverse_group_size" or "none"'
            )
        return reg * scale

    def _get_reg_vector(self, reg):
        """Get the group-wise regularisation coefficients from ``reg``.
        """
        if isinstance(reg, Number):
            reg = [
                self._get_reg_strength(group, reg) for group in self.groups_
            ]
        else:
            reg = list(reg)
        return reg

    @abstractmethod
    def _unregularised_loss(self, X_aug, y, w):  # pragma: nocover
        """The unregularised reconstruction loss.
        """
        raise NotImplementedError

    def _loss(self, X, y, w):
        """The group-lasso regularised loss.

        Parameters
        ----------
        X : np.ndarray
            Data matrix, ``X.shape == (num_datapoints, num_features)``
        y : np.ndarray
            Target vector/matrix, ``y.shape == (num_datapoints, num_targets)``,
            or ``y.shape == (num_datapoints,)``
        w : np.ndarray
            Coefficient vector, ``w.shape == (num_features, num_targets)``,
            or ``w.shape == (num_features,)``
        """
        return self._unregularised_loss(X, y, w) + self._regulariser(w)

    def loss(self, X, y):
        """The group-lasso regularised loss with the current coefficients

        Parameters
        ----------
        X : np.ndarray
            Data matrix, ``X.shape == (num_datapoints, num_features)``
        y : np.ndarray
            Target vector/matrix, ``y.shape == (num_datapoints, num_targets)``,
            or ``y.shape == (num_datapoints,)``
        """
        X_aug = _add_intercept_col(X)
        w = _join_intercept(self.intercept_, self.coef_)
        return self._loss(X_aug, y, w)

    @abstractmethod
    def _estimate_lipschitz(self, X_aug, y):  # pragma: nocover
        """Compute Lipschitz bound for the gradient of the unregularised loss.

        The Lipschitz bound is with respect to the coefficient vector or
        matrix.
        """
        raise NotImplementedError

    @abstractmethod
    def _grad(self, X_aug, y, w):  # pragma: nocover
        """Compute the gradient of the unregularised loss wrt the coefficients.
        """
        raise NotImplementedError

    def _unregularised_gradient(self, X_aug, y, w):
        g = self._grad(X_aug, y, w)
        if not self.fit_intercept:
            g[0] = 0
        return g

    def _scaled_prox(self, w, lipschitz):
        """Apply the proximal map of the scaled regulariser to ``w``.

        The scaling is the inverse lipschitz coefficient.
        """
        b, w_ = _split_intercept(w)
        l1_reg = self.l1_reg
        group_reg_vector = self.group_reg_vector_
        if not self.old_regularisation:
            l1_reg = l1_reg / lipschitz
            group_reg_vector = np.asarray(group_reg_vector) / lipschitz

        w_ = _l1_l2_prox(w_, l1_reg, group_reg_vector, self.groups_)
        return _join_intercept(b, w_)

    def _minimise_loss(self):
        """Use the FISTA algorithm to solve the group lasso regularised loss.
        """
        # Need transition period before the correct regulariser is used without warning
        def callback(x, it_num, previous_x=None):
            X_, y_ = self.subsampler_.subsample(self.X_aug_, self.y_)
            self.subsampler_.update_indices()
            w = x
            previous_w = previous_x

            if self.LOG_LOSSES:
                self.losses_.append(self._loss(X_, y_, w))

                grad_norm = la.norm(
                    self._unregularised_gradient(self.X_aug_, self.y_, w)
                )
                print("\tWeight norm: {wnorm}".format(wnorm=la.norm(w)))
                print("\tGrad: {gnorm}".format(gnorm=grad_norm))
                print(
                    "\tRelative grad: {relnorm}".format(
                        relnorm=grad_norm / la.norm(w)
                    )
                )
                print(
                    "\tLipschitz: {lipschitz}".format(
                        lipschitz=optimiser.lipschitz
                    )
                )

        weights = _join_intercept(self.intercept_, self.coef_)
        optimiser = FISTAProblem(
            self.subsampler_.subsample_apply(
                self._unregularised_loss, self.X_aug_, self.y_
            ),
            self._regulariser,
            self.subsampler_.subsample_apply(
                self._unregularised_gradient, self.X_aug_, self.y_
            ),
            self._scaled_prox,
            self.lipschitz_,
        )
        weights = optimiser.minimise(
            weights, n_iter=self.n_iter, tol=self.tol, callback=callback
        )
        self.lipschitz_ = optimiser.lipschitz
        self.intercept_, self.coef_ = _split_intercept(weights)

    def _check_valid_parameters(self):
        """Check that the input parameters are valid.
        """
        assert all(reg >= 0 for reg in self.group_reg_vector_)
        groups = self.group_ids_
        assert len(self.group_reg_vector_) == len(
            np.unique(groups[groups >= 0])
        )
        assert self.n_iter > 0
        assert self.tol >= 0

    def _prepare_dataset(self, X, y, lipschitz):
        """Ensure that the inputs are valid and prepare them for fit.
        """
        X = check_array(X, accept_sparse="csc")
        y = check_array(y, ensure_2d=False)

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        # Center X for numerical stability
        if not sparse.issparse(X) and self.fit_intercept:
            X_means = X.mean(axis=0, keepdims=True)
            X = X - X_means
        else:
            X_means = np.zeros((1, X.shape[1]))

        # Add the intercept column and compute Lipschitz bound the correct way
        if self.fit_intercept:
            X = _add_intercept_col(X)
            X = check_array(X, accept_sparse="csc")

        if lipschitz is None:
            lipschitz = self._estimate_lipschitz(X, y)

        if not self.fit_intercept:
            X = _add_intercept_col(X)
            X = check_array(X, accept_sparse="csc")

        return X, X_means, y, lipschitz

    def _init_fit(self, X, y, lipschitz):
        """Initialise model and check inputs.
        """
        self.random_state_ = check_random_state(self.random_state)

        check_consistent_length(X, y)
        X, X_means, y, lipschitz = self._prepare_dataset(X, y, lipschitz)
        
        self.subsampler_ = Subsampler(
            X.shape[0], self.subsampling_scheme, self.random_state_
        )

        groups = self.groups
        self.group_ids_ = np.array(_parse_group_iterable(groups))

        self.groups_ = [
            self.group_ids_ == u
            for u in np.unique(self.group_ids_) if u >= 0
        ]
        self.group_reg_vector_ = self._get_reg_vector(self.group_reg)

        self.losses_ = []

        if not self.warm_start or not hasattr(self, "coef_"):
            self.coef_ = np.zeros((X.shape[1] - 1, y.shape[1]))
            self.intercept_ = np.zeros((1, self.coef_.shape[1]))

        self._check_valid_parameters()
        self.X_aug_, self.y_, self.lipschitz_ = X, y, lipschitz
        self._X_means_ = X_means

    def fit(self, X, y, lipschitz=None):
        """Fit a group-lasso regularised linear model.
        """
        self._init_fit(X, y, lipschitz=lipschitz)
        self._minimise_loss()
        self.intercept_ -= (self._X_means_ @ self.coef_).reshape(
            self.intercept_.shape
        )
        return self

    def _compute_scores(self, X):
        w = _join_intercept(self.intercept_, self.coef_)
        if X.shape[1] == self.coef_.shape[0]:
            X = _add_intercept_col(X)

        return X @ w

    @abstractmethod
    def predict(self, X):  # pragma: nocover
        """Predict using the linear model.
        """
        raise NotImplementedError

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)

    @property
    def sparsity_mask(self):
        """A boolean mask indicating whether features are used in prediction.
        """
        warnings.warn(
            "This property is discontinued, use sparsity_mask_ instead of sparsity_mask."
        )
        return self.sparsity_mask_

    def _get_chosen_coef_mask(self, coef_):
        mean_abs_coef = abs(coef_.mean())
        return np.abs(coef_) > 1e-10 * mean_abs_coef

    @property
    def sparsity_mask_(self):
        """A boolean mask indicating whether features are used in prediction.
        """
        coef_ = self.coef_.mean(1)
        return self._get_chosen_coef_mask(coef_)

    @property
    def chosen_groups_(self):
        """A set of the coosen group ids.
        """
        groups = self.group_ids_
        if groups.ndim == 1:
            sparsity_mask = self.sparsity_mask_
        else:
            sparsity_mask = self._get_chosen_coef_mask(self.coef_).ravel()
        groups = groups.ravel()
        # TODO: Add regression test with list input for groups

        return set(np.unique(groups[sparsity_mask]))

    def transform(self, X):
        """Remove columns corresponding to zero-valued coefficients.
        """
        if not hasattr(self, 'coef_'):
            raise NotFittedError
        X = check_array(X, accept_sparse="csc")
        if X.shape[1] != self.coef_.shape[0]:
            raise ValueError(
                "The transformer {} does not raise an error when the number of "
                "features in transform is different from the number of features in "
                "fit.".format(self.__class__.__name__)
            )

        return X[:, self.sparsity_mask_]

    def fit_transform(self, X, y, lipschitz=None):
        """Fit a group lasso model to X and y and remove unused columns from X
        """
        self.fit(X, y, lipschitz)
        return self.transform(X)


def _l2_grad(A, b, x):
    """The gradient of the problem ||Ax - b||^2 wrt x.
    """
    return A.T @ (A @ x - b)


class GroupLasso(BaseGroupLasso, RegressorMixin):

    def __init__(
        self,
        groups=None,
        group_reg=0.05,
        l1_reg=0.05,
        n_iter=100,
        tol=1e-5,
        scale_reg="group_size",
        subsampling_scheme=None,
        fit_intercept=True,
        frobenius_lipschitz=False,
        random_state=None,
        warm_start=False,
        old_regularisation=False,
        supress_warning=False,
    ):
        super().__init__(
            groups=groups,
            l1_reg=l1_reg,
            group_reg=group_reg,
            n_iter=n_iter,
            tol=tol,
            scale_reg=scale_reg,
            subsampling_scheme=subsampling_scheme,
            fit_intercept=fit_intercept,
            random_state=random_state,
            warm_start=warm_start,
            old_regularisation=old_regularisation,
            supress_warning=supress_warning,
        )
        self.frobenius_lipschitz = frobenius_lipschitz

    def fit(self, X, y, lipschitz=None):
        """Fit a group lasso regularised linear regression model.

        Parameters
        ----------
        X : np.ndarray
            Data matrix
        y : np.ndarray
            Target vector or matrix
        lipschitz : float or None [default=None]
            A Lipshitz bound for the mean squared loss with the given
            data and target matrices. If None, this is estimated.
        """
        return super().fit(X, y, lipschitz=lipschitz)

    def predict(self, X):
        """Predict using the linear model.
        """
        if not hasattr(self, 'coef_'):
            raise NotFittedError
        X = check_array(X, accept_sparse="csc")
        scores = self._compute_scores(X)
        if scores.ndim == 2 and scores.shape[1] == 1:
            return scores.reshape(scores.shape[0])
        return scores

    def _unregularised_loss(self, X_aug, y, w):
        MSE = np.sum((X_aug @ w - y) ** 2) / X_aug.shape[0]
        return 0.5 * MSE

    def _grad(self, X_aug, y, w):
        SSE_grad = _l2_grad(X_aug, y, w)
        return SSE_grad / X_aug.shape[0]

    def _estimate_lipschitz(self, X_aug, y):
        num_rows = X_aug.shape[0]
        if self.frobenius_lipschitz:
            if sparse.issparse(X_aug):
                return sparse.linalg.norm(X_aug, "fro") ** 2 / num_rows
            return la.norm(X_aug, "fro") ** 2 / num_rows

        s_max = find_largest_singular_value(
            X_aug,
            subsampling_scheme=self.subsampling_scheme,
            random_state=self.random_state_,
        )
        SSE_lipschitz = 1.5 * s_max ** 2
        return SSE_lipschitz / num_rows