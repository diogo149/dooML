"""Table of Contents
    -modified_huber
    -hinge
    -squared_hinge
    -log
    -squared
    -huber
    -epsilon_insensitive
    -squared_epislon_insensitive
    -alpha_huber
    -absolute
"""

import numpy as np


def modified_huber(p, y):
    """Modified Huber loss for binary classification with y in {-1, 1}; equivalent to quadratically smoothed SVM with gamma = 2
    """
    z = p * y
    loss = -4.0 * z
    idx = z >= -1.0
    loss[idx] = (z[idx] - 1.0) ** 2
    loss[z >= 1.0] = 0.0
    return loss


def hinge(p, y, threshold=1.0):
    """Hinge loss for binary classification tasks with y in {-1,1}

    Parameters
    ----------

    threshold : float > 0.0
        Margin threshold. When threshold=1.0, one gets the loss used by SVM.
        When threshold=0.0, one gets the loss used by the Perceptron.
    """
    z = p * y
    loss = threshold - z
    loss[loss < 0] = 0.0
    return loss


def squared_hinge(p, y, threshold=1.0):
    """Squared Hinge loss for binary classification tasks with y in {-1,1}

    Parameters
    ----------

    threshold : float > 0.0
        Margin threshold. When threshold=1.0, one gets the loss used by
        (quadratically penalized) SVM.
    """
    return hinge(p, y, threshold) ** 2


def log(p, y):
    """Logistic regression loss for binary classification with y in {-1, 1}"""
    z = p * y
    return np.log(1.0 + np.exp(-z))


def squared(p, y):
    """Squared loss traditional used in linear regression."""
    return 0.5 * (p - y) ** 2


def huber(p, y, epsilon=0.1):
    """Huber regression loss

    Variant of the SquaredLoss that is robust to outliers (quadratic near zero,
    linear in for large errors).

    http://en.wikipedia.org/wiki/Huber_Loss_Function
    """
    abs_r = np.abs(p - y)
    loss = 0.5 * abs_r ** 2
    idx = abs_r <= epsilon
    loss[idx] = epsilon * abs_r[idx] - 0.5 * epsilon ** 2
    return loss


def epsilon_insensitive(p, y, epsilon=0.1):
    """Epsilon-Insensitive loss (used by SVR).

    loss = max(0, |y - p| - epsilon)
    """
    loss = np.abs(y - p) - epsilon
    loss[loss < 0.0] = 0.0
    return loss


def squared_epislon_insensitive(p, y, epsilon=0.1):
    """Epsilon-Insensitive loss.

    loss = max(0, |y - p| - epsilon)^2
    """
    return epsilon_insensitive(p, y, epsilon) ** 2


def alpha_huber(p, y, alpha=0.9):
    """ sets the epislon in huber loss equal to a percentile of the residuals
    """
    abs_r = np.abs(p - y)
    loss = 0.5 * abs_r ** 2
    epsilon = np.percentile(loss, alpha * 100)
    idx = abs_r <= epsilon
    loss[idx] = epsilon * abs_r[idx] - 0.5 * epsilon ** 2
    return loss


def absolute(p, y):
    """ absolute value of loss
    """
    return np.abs(p - y)
