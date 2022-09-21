
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binom
import math
import scikit_learn as sk
import os
import pydoc

sns.set(color_codes=True)
sns.set(rc={'figure.figsize':(1.6*8,8)})

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv ('/datasetbookpruchases/ArtHistBooks.csv')

df.head()

df.info()

df.describe()

df_ArtPurchase = df.loc[df['ArtBooks'] > 0]
df_ArtPurchase

x = np.arange(0, 1, 0.01)
L = binom.pmf(k=301, n=1000, p=x)

# compute the denominator in Bayes Theorem (i.e. the normalizing factor) approximating the integral
prior_prob = 1/len(L)
delta_theta = 0.01
D = np.sum(L*prior_prob*delta_theta)

# now compute the probability for each x-value using Bayes Theorem
P= L*prior_prob / D

ax = sns.lineplot(x, P)
ax.set(xlabel='x', ylabel='f(x)', title=f'Probability Density Function for p (constant prior)');











