# Importing the basic libariries 
# We will import the others later this is just to get the analysis started :P

import os
import joblib
import numpy as np
import pandas as pd
import warnings

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/tabular-playground-series-apr-2021/train.csv')
test_df = pd.read_csv('/kaggle/input/tabular-playground-series-apr-2021/test.csv')


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import  StratifiedKFold
from sklearn import metric