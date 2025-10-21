#!/usr/bin/env python
# coding: utf-8
"""
Wet Days Calculation
====================
Calculates the number of wet days (daily precipitation >1mm) for DHS mortality data

Reference:
He, C., Zhu, Y., Guo, Y., et al. (2025). Rainfall variability and under-five 
child mortality in 59 low- and middle-income countries. Nature Water, 3, 881-889.
https://doi.org/10.1038/s44221-025-00478-9

Author: Cheng He, Yixiang Zhu
"""

import pandas as pd
from multiprocessing import Pool
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================
BASE_DIR = Path('..../DATA')
INPUT_DIR = BASE_DIR / 'pf_result'
OUTPUT_DIR = BASE_DIR / 'pf_wetdays_results'
CITY_LIST_DIR = BASE_DIR / 'city_list'
PRECIP_DIR = Path('.../precip_csv')

WET_DAY_THRESHOLD = 1.0  # mm, can be changed to 0.5 or 2.0 for sensitivity analysis
MIN_YEAR = 1981
N_MONTHS = 12
N_PROCESSES = 50


# ============================================================================
# Core Functions
# ============================================================================
def calc_wet_days(target_date, precip_data, threshold=WET_DAY_THRESHOLD):
    """
    Calculate number of wet days for one month.
    
    Parameters
    ----------
    target_date : pd.Timestamp
        Target month date
    precip_data : pd.DataFrame
        Daily precipitation data
    threshold : float
        Wet day threshold in mm (default: 1.0mm)
    
    Returns
    -------
    int
        Number of wet days in the month
    """
    year, month = target_date.year, target_date.month
    
    # Filter data for target month
    monthly = precip_data[
        (precip_data['time'].dt.year == year) & 
        (precip_data['time'].dt.month == month)
    ]
    
    # Count days exceeding threshold
    wet_days = (monthly['tp'] > threshold).sum()
    
    return wet_days


def load_precip_data(dhsid, survey_id):
    """Load precipitation data for location"""
    survey_clean = survey_id.replace("Cote_d'Ivoire", "Cote_dIvoire")
    
    # Get city ID
    city_map = pd.read_csv(CITY_LIST_DIR / f'{survey_id}.csv')
    city_id = str(int(city_map[city_map['DHSID'] == dhsid]['city_id'].values[0]))
    
    # Load precipitation data
    precip = pd.read_csv(PRECIP_DIR / survey_clean / f'{city_id}.csv')
    precip['time'] = pd.to_datetime(precip['time'])
    
    return precip


def process_file(filename):
    """Process single DHS file"""
    try:
        # Load data
        df = pd.read_csv(INPUT_DIR / filename)
        
        # Prepare dates
        df['death_date_lag_0'] = pd.to_datetime(df['death_date_lag_0'])
        df = df[df['death_date_lag_0'].dt.year >= MIN_YEAR].reset_index(drop=True)
        
        # Initialize result columns
        for m in range(1, N_MONTHS + 1):
            df[f'wet_days_{m}'] = 0
        
        # Process each row
        for idx in range(len(df)):
            dhsid = df['DHSID'].iloc[idx]
            survey = df['DHS_survey'].iloc[idx]
            base_date = df['death_date_lag_0'].iloc[idx]
            
            # Load precipitation data
            precip = load_precip_data(dhsid, survey)
            
            # Calculate wet days for each month
            for m in range(1, N_MONTHS + 1):
                target_date = base_date - pd.DateOffset(months=m)
                wet_days = calc_wet_days(target_date, precip)
                df.at[idx, f'wet_days_{m}'] = wet_days
            
            # Progress indicator
            if (idx + 1) % 100 == 0:
                print(f'{filename}: {idx + 1}/{len(df)} rows')
        
        # Calculate annual total
        df['wet_days_annual'] = df[
            [f'wet_days_{m}' for m in range(1, N_MONTHS + 1)]
        ].sum(axis=1)
        
        # Save results
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
    print(f'Using wet day threshold: {WET_DAY_THRESHOLD}mm\n')
    
    with Pool(N_PROCESSES) as pool:
        results = pool.map(process_file, [f.name for f in files])
    
    success = sum(1 for r in results if r)
    print(f'\nCompleted: {success}/{len(files)} files')


if __name__ == '__main__':
    main()
