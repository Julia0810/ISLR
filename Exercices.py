# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 16:01:01 2020

@author: La douce
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 09:38:28 2020

@author: la douce
"""

##########   Ex 3.8   ##########

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

Auto = pd.read_csv("C:/Users/fred/Documents/julia/ISLR/Auto.csv")

y = Auto["mpg"].values.reshape(-1,1)
x1 = Auto["horsepower"].values.reshape(-1,1)
x2 = sm.add_constant(x1) 

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x1, y, s = 2)

results = sm.OLS(y,x2).fit()
results.summary()
results.conf_int(0.01) #confidence interval of coef(s) with 99% certainty

ax.plot([0, 230], [results.params[0], 230*results.params[1] + results.params[0]])

mpg98 = 98*results.params[1] + results.params[0]

predictions = results.get_prediction([1,98])
predictions.summary_frame(alpha=0.05)

##########   Ex 3.9   ##########

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

auto = pd.read_csv("C:/Users/fred/Documents/julia/ISLR/Auto.csv")
auto.drop(auto.columns[0],1, inplace=True)

axes = pd.plotting.scatter_matrix(auto, figsize=(10,10))
plt.tight_layout()

auto.corr()

y = auto.iloc[:,0]
x = auto.iloc[:,1:8]
X = sm.add_constant(x) 

model = sm.OLS(y, X)
result = model.fit()
result.summary()

##########   Ex 3.10   ##########

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

carseats = pd.read_csv("C:/Users/fred/Documents/julia/ISLR/Carseats.csv")

carseats["Urban"] = carseats["Urban"].eq('Yes').mul(1)
carseats["US"] = carseats["US"].eq('Yes').mul(1)

y = carseats.iloc[:,1]
x = carseats.iloc[:,[6,-1,-2]]
X = sm.add_constant(x) 

model = sm.OLS(y, X)
result = model.fit()
result.summary()

##########   Ex 3.11   ##########







































