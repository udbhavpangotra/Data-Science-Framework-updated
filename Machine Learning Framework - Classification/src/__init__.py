# Importing the basic libariries 
# We will import the others later this is just to get the analysis started :P

import os
import joblib
import numpy as np
import pandas as pd
import warnings

# visualization
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

# models
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import  StratifiedKFold
from sklearn import metric



pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')

# ENTER THE LOCATION OF THE TRAIN AND THE TEST FILE

train_data_location = ""  
test_data_location = ""

