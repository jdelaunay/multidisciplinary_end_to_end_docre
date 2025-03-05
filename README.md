# Multidisciplinary End-to-End Document-level Relation Extraction from Scientific Literature

## Description
In this repo, we provide the code and dataset for performing End-to-End Document-level Relation from scientific abstracts about coastal areas.

## Requirements
Please install all the necessary libraries noted in [requirements.txt](./requirements.txt) using this command:

```
pip install -r requirements.txt
```

## Data
The experiments were conducted on the End-to-End version of DocRED, ReDocRED, and our novel dataset CoastRED. Datasets are stored in the [data](./data/) directory.

## Training
To train the model on CoastRED for End-to-End DocRE, use this command:

```
python train.py configs/config_coastred.json
```

For isolated task training, set the coefficients of the task for which you want to train to 1, and the others to 0.

## Testing
To train the model on CoastRED, use this command:

```
python test.py configs/config_coastred.json
```
In the configuration file, respect the coefficients used in training, and provide the checkpoint in `"pretrained_weights"`.