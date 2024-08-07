# Configura visualización en consola

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 800)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 1000)


# Fija semilla aleatoria
import numpy as np

np.random.seed(1234)