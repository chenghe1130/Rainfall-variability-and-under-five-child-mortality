#!/usr/bin/env python
# coding: utf-8
"""
Extreme Precipitation Metrics Calculation
==========================================
Calculates 99.9th percentile extreme precipitation indicators for DHS mortality data

Reference:
He, C., Zhu, Y., Guo, Y., et al. (2025). Rainfall variability and under-five 
child mortality in 59 low- and middle-income countries. Nature Water, 3, 881-889.
https://doi.org/10.1038/s44221-025-00478-9

Author: Cheng He, Yixiang Zhu
License: MIT
"""

import pandas as pd
from multiprocessing import Pool
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================
BASE_DIR = Path('/d2/home/user7/extreme_rainfall/DHS_RAINFALL/DATA')
INPUT_DIR = BASE_DIR / 'pf_result'
OUTPUT_DIR = BASE_DIR / 'pf_p999_results'
CITY_LIST_DIR = BASE_DIR / 'city_list'
PRECIP_DIR = Path('/d2/public/DHS/DHS_exposure_matching/met_csv/precip_csv')
THRESHOLD_FILE = BASE_DIR / 'all_points_dhs_prep.csv'

PERCENTILE = 99.9
MIN_YEAR = 1981
N_MONTHS = 12
N_PROCESSES = 50


# ============================================================================
# Core Functions
# ============================================================================
def calc_extreme_precip(target_date, precip_data, threshold, avg_precip):
    """
    Calculate extreme precipitation metrics for one month.
    
    Parameters
    ----------
    target_date : pd.Timestamp
        Target month date
    precip_data : pd.DataFrame
        Daily precipitation data
    threshold : float
        Extreme precipitation threshold (99.9th percentile)
    avg_precip : float
        Average precipitation for wet days
    
    Returns
    -------
    tuple
        (n_days, excess_precip, relative_intensity)
    """
    year, month = target_date.year, target_date.month
    
    monthly = precip_data[
        (precip_data['time'].dt.year == year) & 
        (precip_data['time'].dt.month == month)
    ]
    extreme = monthly[monthly['tp'] > threshold]
    
    if len(extreme) == 0:
        return 0, 0.0, 0.0
    
    n_days = len(extreme)
    excess = extreme['tp'].sum() - threshold * n_days
    intensity = excess / avg_precip if avg_precip > 0 else 0.0
    
    return n_days, excess, intensity


def load_precip_data(dhsid, survey_id, threshold_df):
    """Load precipitation data and threshold for location"""
    survey_clean = survey_id.replace("Cote_d'Ivoire", "Cote_dIvoire")
    
    city_map = pd.read_csv(CITY_LIST_DIR / f'{survey_id}.csv')
    city_id = str(int(city_map[city_map['DHSID'] == dhsid]['city_id'].values[0]))
    
    precip = pd.read_csv(PRECIP_DIR / survey_clean / f'{city_id}.csv')
    precip['time'] = pd.to_datetime(precip['time'])
    
    threshold = float(threshold_df[threshold_df['DHSID'] == dhsid]['p999'].values[0])
    avg_precip = precip[precip['tp'] > 0]['tp'].mean()
    
    return precip, threshold, avg_precip


def process_file(filename):
    """Process single DHS file"""
    try:
        df = pd.read_csv(INPUT_DIR / filename)
        threshold_df = pd.read_csv(THRESHOLD_FILE)
        
        df['death_date_lag_0'] = pd.to_datetime(df['death_date_lag_0'])
        df = df[df['death_date_lag_0'].dt.year >= MIN_YEAR].reset_index(drop=True)
        
        metrics = ['days', 'excess', 'intensity']
        for metric in metrics:
            for m in range(1, N_MONTHS + 1):
                df[f'p999_{metric}_{m}'] = 0.0
        
        for idx in range(len(df)):
            dhsid = df['DHSID'].iloc[idx]
            survey = df['DHS_survey'].iloc[idx]
            base_date = df['death_date_lag_0'].iloc[idx]
            
            precip, threshold, avg = load_precip_data(dhsid, survey, threshold_df)
            
            for m in range(1, N_MONTHS + 1):
                target_date = base_date - pd.DateOffset(months=m)
                n_days, excess, intensity = calc_extreme_precip(
                    target_date, precip, threshold, avg
                )
                df.at[idx, f'p999_days_{m}'] = n_days
                df.at[idx, f'p999_excess_{m}'] = excess
                df.at[idx, f'p999_intensity_{m}'] = intensity
        
        df['p999_annual_days'] = df[
            [f'p999_days_{m}' for m in range(1, N_MONTHS + 1)]
        ].sum(axis=1)
        df['p999_annual_excess'] = df[
            [f'p999_excess_{m}' for m in range(1, N_MONTHS + 1)]
        ].sum(axis=1)
        
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_DIR / filename, index=False)
        
        print(f'✓ {filename}')
        return filename
        
    except Exception as e:
        print(f'✗ {filename}: {e}')
        return None


def main():
    """Run parallel processing"""
    files = [f for f in INPUT_DIR.glob('*.csv')]
    print(f'Processing {len(files)} files with {N_PROCESSES} processes...')
    print(f'Using {PERCENTILE}th percentile threshold\n')
    
    with Pool(N_PROCESSES) as pool:
        results = pool.map(process_file, [f.name for f in files])
    
    success = sum(1 for r in results if r)
    print(f'\nCompleted: {success}/{len(files)} files')


if __name__ == '__main__':
    main()
