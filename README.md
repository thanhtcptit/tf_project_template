# Machine Learning Project Template

### Requirements to use this template:
-----------
 - Python >= 3.6
 - [Cookiecutter Python package](http://cookiecutter.readthedocs.org/en/latest/installation.html) >= 1.4.0: This can be installed with pip by or conda depending on how you manage your Python packages:

``` bash
$ pip install cookiecutter
```

or

``` bash
$ conda config --add channels conda-forge
$ conda install cookiecutter
```


### To start a new project, run:
------------

    cookiecutter https://github.com/thanhtcptit/ml_template




### The resulting directory structure
------------

The directory structure of your new project looks like this: 

```
├── configs                     <- Store experiment config files
├── data
│   ├── external                <- Data from third party sources
│   ├── interim                 <- Intermediate data that has been transformed
│   ├── processed               <- The final, canonical data sets for modeling
│   └── raw                     <- The original, immutable data dump
├── train_logs                  <- Trained and serialized models, model predictions, or model summaries  
├── notebooks                   <- Jupyter notebooks
├── references                  <- Data dictionaries, manuals, and all other explanatory materials
├── reports                     <- Generated analysis as HTML, PDF, LaTeX, etc
├── resources                   <- Other resources for the project
├── src                         <- Source code for use in this project.
│   ├── data                    <- Scripts to download or generate data
│   ├── features                <- Scripts to turn raw data into features for modeling
│   ├── models                  <- Scripts to define models
│   ├── utils                   <- Scripts to define helper function
│   ├── visualization           <- Scripts to create exploratory and results oriented visualizations
│   ├── __init__.py
│   ├── evaluate.py
│   └── train.py
├── README.md                   <- The top-level README for developers using this project
├── requirements.txt            <- The requirements file for reproducing the analysis environment
├── run.py                      <- Script to run tasks 
└── setup.py                    <- Makes project pip installable (pip install -e .) so src can be imported
```

### Installing development requirements
------------

    pip install -r requirements.txt
