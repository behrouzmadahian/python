import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import numpy as np
from statsmodels.graphics.gofplots import  qqplot
from matplotlib import  pyplot as plt
data = pd.DataFrame({'len': np.random.rand(400),
                     'supp': [1]*100+[2] * 300,
                     'dose': ['one']*200+['two']*200})


formula = 'len ~ C(supp) + C(dose) + C(supp):C(dose)'
model = ols(formula, data).fit()
aov_table = anova_lm(model, typ=2)
print(aov_table)

res = model.resid
fig = qqplot(res, line='s')
plt.show()
