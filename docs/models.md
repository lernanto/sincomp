# Neural models

`sincomp.nn` provides TensorFlow-based building blocks for predicting
pronunciation features from dialect and character inputs. TensorFlow is
optional; install the `models` extra before importing these classes:

```shell
python -m pip install "sincomp[models]"
```

## Encode categorical inputs

`Processor` is initialized with one vocabulary per dialect input,
character input, and prediction target. It maps known values to integer
IDs and reserves ID `0` for missing or unknown values.

```python
import numpy as np
from sincomp.nn import BilinearEncoder, Processor

processor = Processor(
    dialect_vocabs=[["dialect-a", "dialect-b"]],
    char_vocabs=[["c1", "c2"]],
    target_vocabs=[["p", "t"], ["a", "i"], ["1", "2"]],
)

dialect_ids, character_ids, target_ids = processor(
    np.array([["dialect-a"]]),
    np.array([["c1"]]),
    np.array([["p", "a", "1"]]),
)

model = BilinearEncoder(
    processor.dialect_vocab_sizes,
    processor.char_vocab_sizes,
    processor.target_vocab_sizes,
)
predicted_ids = model.predict(dialect_ids, character_ids)
```

Each input array has one column per corresponding vocabulary. The target
array has one column per feature to predict. `predict()` returns integer
target IDs; `predict_proba()` returns one probability matrix per target.

## Train an encoder

`EncoderBase` provides shared encoding, decoding, loss, evaluation, and
training methods. `BilinearEncoder` uses a dialect-conditioned bilinear
transformation of the character embedding; set `residual=True` to add
the input character embedding to the transformed output.

For training, construct a `tf.data.Dataset` whose elements are
`((dialect_ids, character_ids), target_ids)` and pass it with a TensorFlow
optimizer:

```python
import tensorflow as tf

train_data = tf.data.Dataset.from_tensor_slices(
    ((dialect_ids, character_ids), target_ids)
)
model.fit(tf.optimizers.Adam(), train_data, epochs=20)
```

The model's `fit()` method also accepts validation data, target weights,
batch size, checkpoint directory, and TensorBoard log directory. Review
vocabulary coverage and train/validation splits before evaluating
predictions.

See [`sincomp.nn`](../src/sincomp/nn.py) for the public model classes and
their docstrings. The TensorFlow implementation lives in
[`sincomp._internal.models`](../src/sincomp/_internal/models.py).
