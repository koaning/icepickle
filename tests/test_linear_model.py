import pytest
import pathlib

import numpy as np
from sklearn.linear_model import (
    SGDClassifier,
    LogisticRegression,
    PassiveAggressiveClassifier,
    LinearRegression,
    SGDRegressor,
    PassiveAggressiveRegressor,
)
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.base import is_classifier

from icepickle.linear_model import load_coefficients, save_coefficients


def compare_models(model1, model2, tmpdir):
    clf = model1()
    pipe = make_pipeline(
        make_union(
            HashingVectorizer(), HashingVectorizer(ngram_range=(2, 3), analyzer="char")
        ),
        clf,
    )

    X = [
        "i really like this post",
        "thanks for that comment",
        "i enjoy this friendly forum",
        "this is a bad post",
        "i dislike this article",
        "this is not well written",
    ]

    y = np.array([1, 1, 1, 0, 0, 0])

    pipe.fit(X, y)

    if is_classifier(clf):
        assert np.all(pipe.predict(X) == y)

    # Here we create in the new pipeline.
    clf_new = model2()
    pipe_new = make_pipeline(
        make_union(
            HashingVectorizer(), HashingVectorizer(ngram_range=(2, 3), analyzer="char")
        ),
        clf_new,
    )
    path = pathlib.Path(tmpdir, "coefs.h5")
    save_coefficients(clf, path)
    load_coefficients(clf_new, path)
    assert np.all(clf.intercept_ == clf_new.intercept_)
    assert np.all(clf.coef_ == clf_new.coef_)
    if is_classifier(clf_new):
        assert np.all(clf.classes_ == clf_new.classes_)
        assert np.all(pipe_new.predict(X) == y)


@pytest.mark.parametrize(
    "clf_train", [LogisticRegression, SGDClassifier, PassiveAggressiveClassifier]
)
@pytest.mark.parametrize(
    "clf_target", [LogisticRegression, SGDClassifier, PassiveAggressiveClassifier]
)
def test_load_save_clf(clf_train, clf_target, tmpdir):
    """
    Ensure that we can save/load vectors.
    """
    compare_models(clf_train, clf_target, tmpdir=tmpdir)


@pytest.mark.parametrize(
    "reg_train", [LinearRegression, SGDRegressor, PassiveAggressiveRegressor]
)
@pytest.mark.parametrize(
    "reg_target", [LinearRegression, SGDRegressor, PassiveAggressiveRegressor]
)
def test_load_save_reg(reg_train, reg_target, tmpdir):
    """
    Ensure that we can save/load vectors.
    """
    compare_models(reg_train, reg_target, tmpdir=tmpdir)
