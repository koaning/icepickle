<img src="icepickle.png" width=210 align="right">

# icepickle

> It's a cooler way to store simple linear models.

The goal of **icepickle** is to allow a safe way to serialize and deserialize linear
scikit-learn models. Not only is this much safer, but it also allows for an interesting
finetuning pattern that does not require a GPU.

## Installation

You can install everything with `pip`:

```
python -m pip install icepickle
```

## Usage

Let's say that you've gotten a linear model from scikit-learn trained on a dataset.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_wine

X, y = load_wine(return_X_y=True)

mod = LogisticRegression()
mod.fit(X, y)
```

Then you *could* use a `pickle` to save the model.

```python
from joblib import dump, load

# You can save the classifier.
dump(clf, 'classifier.joblib')

# You can load it too.
clf_reloaded = load('classifier.joblib')
```

But this is unsafe. The scikit-learn documentations warns about the [security concerns](https://scikit-learn.org/stable/modules/model_persistence.html#security-maintainability-limitations) but also about potential compatibility issues. The goal of this package is to offer a safe alternative to pickling for simple linear models. The coefficients will be saved in a `.h5` file and an be loaded into a new regression model later.

```python
from icepickle.linear_model import save_coefficients, load_coefficients

# You can save the classifier.
save_coefficients(clf, 'classifier.h5')

# You can load it too.
clf_reloaded = load_coefficients('classifier.h5')
```

This is a lot safer and there's plenty of use-cases that could be handled this way.
## Finetuning

Assuming that you use a stateless featurizer in your pipeline, such as [HashingVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html#sklearn.feature_extraction.text.HashingVectorizer) or language models from [whatlies](https://koaning.github.io/whatlies/api/language/universal_sentence/), you choose to pre-train your scikit-learn model beforehand and fine-tune it later using models that offer the `.partial_fit()`-api. If you're unfamiliar with this api, you might appreciate [this course on calmcode](https://calmcode.io/partial_fit/introduction.html).

The script below demonstrates how the fine-tuning might be used.


```python
import pandas as pd
from sklearn.linear_model import SGDClassifier, LogisticRegression

url = "https://github.com/koaning/icepickle"
df = pd.read_csv(url)
```

<details>
  <summary>Supported Models</summary>

We unit test against the following models.

```python
from sklearn.linear_model import (
    SGDClassifier,
    SGDRegressor,
    LinearRegression,
    LogisticRegression,
    PassiveAggressiveClassifier,
    PassiveAggressiveRegressor,
)
```
</details>