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

clf = LogisticRegression()
clf.fit(X, y)
```

Then you *could* use a `pickle` to save the model.

```python
from joblib import dump, load

# You can save the classifier.
dump(clf, 'classifier.joblib')

# You can load it too.
clf_reloaded = load('classifier.joblib')
```

But this is [unsafe](https://www.youtube.com/watch?v=jwzeJU_62IQ&ab_channel=PwnFunction). The scikit-learn documentations even warns about the [security concerns and compatibility issues](https://scikit-learn.org/stable/modules/model_persistence.html#security-maintainability-limitations). The goal of this package is to offer a safe alternative to pickling for simple linear models. The coefficients will be saved in a `.h5` file and can be loaded into a new regression model later.

```python
from icepickle.linear_model import save_coefficients, load_coefficients

# You can save the classifier.
save_coefficients(clf, 'classifier.h5')

# You can create a new model, with new hyperparams.
clf_reloaded = LogisticRegression()

# Load the previously trained weights in.
load_coefficients(clf_reloaded, 'classifier.h5')
```

This is a lot safer and there's plenty of use-cases that could be handled this way.
## Finetuning

Assuming that you use a stateless featurizer in your pipeline, such as [HashingVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html#sklearn.feature_extraction.text.HashingVectorizer) or language models from [whatlies](https://koaning.github.io/whatlies/api/language/universal_sentence/), you choose to pre-train your scikit-learn model beforehand and fine-tune it later using models that offer the `.partial_fit()`-api. If you're unfamiliar with this api, you might appreciate [this course on calmcode](https://calmcode.io/partial_fit/introduction.html).

The script below demonstrates how the fine-tuning might be used.


```python
import pandas as pd
from sklearn.linear_model import SGDClassifier, LogisticRegression
from icepickle.linear_model import save_coefficients, load_coefficients


url = "https://raw.githubusercontent.com/koaning/icepickle/main/datasets/imdb_subset.csv"
df = pd.read_csv(url)
X, y = list(df['text']), df['label']

# Train a pre-trained model.
pretrained = LogisticRegression().fit(X, y)
save_coefficients(pretrained, 'pretrained.h5')

# Create a new model using pre-trained weights.
finetuned = SGDClassifier()
load_coefficients(finetuned, 'pretrained.h5')
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