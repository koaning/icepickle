import h5py
from sklearn.base import is_classifier


def save_coefficients(model, filename):
    """Save the coefficients of a linear model into a .h5 file."""
    with h5py.File(filename, "w") as hf:
        hf.create_dataset("coef", data=model.coef_)
        hf.create_dataset("intercept", data=model.intercept_)
        if is_classifier(model):
            hf.create_dataset("classes", data=model.classes_)


def load_coefficients(model, filename):
    """Attach the saved coefficients to a linear model."""
    with h5py.File(filename, "r") as hf:
        coef = hf["coef"][:]
        # Sometimes the intercept is stored as a scalar
        # hence, we check how to retreive from h5.
        if len(hf["intercept"].shape) != 0:
            intercept = hf["intercept"][:]
        else:
            # https://github.com/h5py/h5py/issues/1779
            intercept = hf["intercept"][()]
        if is_classifier(model):
            classes = hf["classes"][:]
    model.coef_ = coef
    model.intercept_ = intercept
    if is_classifier(model):
        model.classes_ = classes
