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

```

```

Then you *could* use a `pickle` to save the model. But that's not safe.

```

```

If you're eager to learn why, you may appreciate this resource. The goal of this package is to offer a safe alternative. The coefficients will be saved in a `.h5` file and an be loaded into a new regression model later.



## Finetuning

Assuming that you use a stateless featurizer in your pipeline, such as [HashingVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html#sklearn.feature_extraction.text.HashingVectorizer) or language models from [whatlies](https://koaning.github.io/whatlies/api/language/universal_sentence/), you choose to pre-train your scikit-learn model beforehand and fine-tune it later using models that offer the `.partial_fit()`-api. These would include the [SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier) and [PassiveAgressiveClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html#sklearn.linear_model.PassiveAggressiveClassifier).

```

```