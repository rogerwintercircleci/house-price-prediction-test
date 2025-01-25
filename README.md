# Simple ML House Price Prediction Project

## Clone the repository

```bash
git clone https://github.com/CIRCLECI-GWP/house-price-prediction
cd house-price-prediction
```

This repository is meant to be cloned as part of the steps of following the related CircleCI blog tutorial.

The `circleci` branch has the CircleCI config included that aligns with the tutorial when completed.

## Set up the virtual environment

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Scripts

### Preprocess the data

```bash
python preprocess.py
```

### Train the model

```bash
python train.py
```

### Evaluate the model

```bash
python evaluate.py
```
