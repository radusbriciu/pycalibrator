import os
import pandas as pd
from pathlib import Path
from model_settings import ms
from joblib import Parallel, delayed


def extract_otms(file_dir):
    raw = pd.read_csv(file_dir)
    raw = raw[
        [
            'underlying_symbol', 'quote_datetime', 'sequence_number', 'root',
            'expiration', 'strike', 'option_type', 'trade_size',
            'trade_price',
            'best_bid', 'best_ask', 'trade_iv', 'trade_delta', 'underlying_bid',
        ]
    ]
    raw = raw.rename(columns={'strike':'strike_price','option_type':'w'})
    df = raw.copy()
    df = df.rename(columns={'underlying_bid':'spot_price'})
    df['quote_datetime'] = pd.to_datetime(df['quote_datetime'])
    df['expiration'] = pd.to_datetime(df['expiration'],format='%Y-%m-%d')
    df['days_to_maturity'] = (df['expiration'] - df['quote_datetime']) / pd.Timedelta(days=1)
    df['days_to_maturity'] = df['days_to_maturity'].astype(int)
    df = df[df['days_to_maturity']>0]
    df = df[df['spot_price']>0]
    df = df[df['strike_price']>0]
    df = df[df['trade_iv']>0].copy()
    df['w'] = df['w'].replace({'C': 'call', 'P': 'put'})
    df = df[['quote_datetime', 'strike_price', 'w', 'trade_size', 'trade_price','trade_iv', 'spot_price','days_to_maturity']]
    df['moneyness'] = ms.df_moneyness(df)
    df = df[df['moneyness']<0]
    df = df.drop(columns='moneyness').dropna()
    times = df['quote_datetime'].drop_duplicates().sort_values()
    try:
        df = df[df['quote_datetime'].isin(times)].copy().reset_index(drop=True)
        otm = df.sort_values(by='quote_datetime',ascending=True).reset_index(drop=True)
        t = times.iloc[-1].strftime('%Y-%m-%d')
        if not os.path.exists('otm'):
            os.mkdir('otm')
        otm.to_csv(os.path.join('otm',f'cboe_vix_otm_{t}.csv'))
    except Exception as e:
        print(e)
        pass


def list_csvs(path_object):
    ms.find_root(Path())
    datadir = os.path.join(ms.root,path_object)
    files = [os.path.join(datadir,f) for f in os.listdir(datadir) if f.endswith('.csv') and f.startswith('UnderlyingOptions')]
    return files


files = list_csvs(ms.cboe_vix_trades)

Parallel()(delayed(extract_otms)(f) for f in files)