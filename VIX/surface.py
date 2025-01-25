import os
import pandas as pd
import numpy as np
from pathlib import Path
from plotters import ScatterSurface

path = os.path.join(Path().resolve(),'otm')
files = [os.path.join(path,f) for f in os.listdir(path)]
counter = 0
df = pd.concat([pd.read_csv(f) for f in files])
df['spot_price'] = np.round(df['spot_price'],0)
S = df['spot_price'].drop_duplicates()
df['quote_datetime'] = pd.to_datetime(df['quote_datetime'],format='mixed')
for s in S:
    vols = df[df['spot_price']==s].copy()
    vols['date'] = vols['quote_datetime'].dt.floor('D')
    dates = vols['date'].drop_duplicates().sort_values().reset_index(drop=True)
    last_date = dates.iloc[-1]
    vols = vols[vols['date']==last_date]
    if vols.shape[0]>counter:
        surf = vols.pivot_table(index='strike_price',columns='days_to_maturity',values='trade_iv',aggfunc='last')
        spot = s
        counter += vols.shape[0]
        date = last_date


ScatterSurface(surf.index,surf.columns,surf.values)
