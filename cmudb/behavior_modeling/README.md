# Behavior Modeling

This document details the behavior modeling interface.
## Training (`train.py`)

Trains and serializes models
#### Model Data
- model_name is model variant and training timestamp

```
model_name/ 
    <model_name>.pkl # serialized model
    <model_name>.json # model metadata
    evaluations/ # model evaluations on train/test data
```

#### Model Metadata
- Model name - just a timestamp and model variant
- Training data experiment name

## Evaluation (`evaluate.py`)

Evaluation runs inference on labeled data then computes and serializes performance metrics.

```
evaluate(model, eval_dataset)
Creates a new evaluation for <model, eval_dataset> pair.  Overwrites any existing such evaluation.
```

## Inference (`inference.py`)

Inference runs on unlabeled data and serializes results.

```
inference(model, X)
TODO: flesh this out
```
