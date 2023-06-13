import pandas as pd
import scipy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('elonmusk.csv')

del df['Datetime']
del df['Username']
del df['Tweet Id']

df = df[~df['Text'].str.contains('@')]
df = df[~df['Text'].str.contains('https')]
df = df[~df['Text'].str.contains('http')]

print(df.head())

