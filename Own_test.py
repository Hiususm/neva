# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 10:54:34 2023

@author: Diego Hager
"""

import neva

# parsing data
bsys, params = neva.parse_csv('data/balance_sheets.csv', 'data/exposures_table.csv')

# Geometric Browianian Motion on external assets, whose volatility is
# estimated via the volatility of equities.
sigma_equity = [float(params[bnk]['sigma_equity']) for bnk in params]
bsys = neva.BankingSystemGBMse.with_sigma_equity(bsys, sigma_equity)

# storing initial equity
equity_start = bsys.get_equity()

# shocks to initial equity: 50%
equity_delta = equity_start[:]
equity_delta = [e * 0.5 for e in equity_start]

# running ex-ante Black and Cox, as in [2] 
# with recovery rate equal to 60%
recovery_rate = [0.6 for _ in bsys] 
neva.shock_and_solve(bsys, equity_delta, 'exante_furfine_merton_gbm', 
                     solve_assets=False, recovery_rate=recovery_rate)

# reading equities after one round and after all rounds  
equity_direct = bsys.history[1]
equity_final = bsys.history[-1]

#%%[Test stuff]

# Need to convert the list in the banking system to an array.
# Maturity does nothing ?
# Exposures have to sum to <1

import matplotlib.pyplot as plt
import numpy as np

Names = [o for o,i in params.items()]
for i in range(np.array(bsys.history).shape[1]):
    plt.plot(np.array(bsys.history)[:,i], label = Names[i])
plt.legend()
plt.grid()
plt.show()