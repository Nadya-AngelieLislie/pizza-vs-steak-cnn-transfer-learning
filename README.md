# Pizza vs Steak: CNN From Scratch vs Transfer Learning

> Binary food image classification comparing a Tiny VGG with ResNet50, EfficientNetB3 and InceptionV3.

## Overview

- Deep Learning and Optimization Method group-learning assignment (GSLC 2025): build one from-scratch CNN and test three pre-trained CNNs on Pizza vs Steak images.
- Also explains why EfficientNetB3 needs its own `preprocess_input` step.

## Data / Scope

- Balanced dataset: 1,200 training, 300 validation and 500 test images (50% pizza, 50% steak).

## Workflow

- Explored the data and label balance, then built an augmented image pipeline.
- Model 1: Tiny VGG trained from scratch.
- Models 2-4: pre-trained ResNet50, EfficientNetB3 and InceptionV3 with a new classification head.
- Evaluated each model on the test set with accuracy, precision, recall and F1 per class, and checked sample predictions.

## Results

- Test accuracy: Tiny VGG (from scratch) 87.4%, ResNet50 68.0%, EfficientNetB3 99.6%, InceptionV3 99.2%.
- Pre-trained EfficientNetB3 and InceptionV3 clearly beat the from-scratch model; ResNet50 underperformed it in this setup.
- Using the matching `preprocess_input` for EfficientNetB3 is part of the experiment.

## Tech Stack

Python, TensorFlow/Keras, Tiny VGG, ResNet50, EfficientNetB3, InceptionV3

## Repository Structure

```
notebook/    Full notebook with code, outputs and written interpretation
figures/     Key charts exported from the notebook
requirements.txt
```

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebook/GSLC_Pizza_vs_Steak_CNN.ipynb
```

The notebook was written in Google Colab; the dataset is not included, so upload it to the Colab session (or edit the path in the first data-loading cell).

## Team

Course assignment (Deep Learning and Optimization Method, BINUS University); notebook by Nadya Angelie Lislie.

## Notes

- Student ID numbers were removed from the notebook text.
- The Pizza vs Steak dataset (a subset of Food-101) is downloaded inside the notebook.

## Author

Nadya Angelie Lislie, BINUS University (Computer Science and Statistics). [LinkedIn](https://www.linkedin.com/in/nadya-angelie-lislie/) | [GitHub](https://github.com/Nadya-AngelieLislie)

## Experiment Tracking

`src/experiment_tracking.py` provides an optional MLflow wrapper (`track_run`) for logging parameters, metrics and artifacts; it does nothing when MLflow is not installed. Run `mlflow ui` after logging runs to compare experiments.

## Continuous Integration

GitHub Actions (`.github/workflows/ci.yml`) runs `ruff` and `pytest` repository-health checks on every push; run them locally with `make lint test`.
