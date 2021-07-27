# Importing the basic libariries 
# We will import the others later this is just to get the analysis started :P
!pip install texthero
import pandas as pd
import numpy as np
import texthero as hero
from texthero import preprocessing as ppe
from texthero import visualization as viz
import spacy
from spacy import displacy
import re
import warnings
warnings.filterwarnings("ignore")
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_score, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier

train_df = pd.read_csv('../input/nlp-getting-started/train.csv')
train_df.head()

test_df = pd.read_csv('../input/nlp-getting-started/test.csv')
test_df.head()

