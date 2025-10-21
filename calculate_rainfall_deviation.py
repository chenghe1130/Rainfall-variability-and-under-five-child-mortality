#!/usr/bin/env python
# coding: utf-8
"""
Monthly Rainfall Deviation (RSD) Calculation
=============================================
Calculates standardized monthly rainfall deviations for DHS mortality data

This script implements the RSD formula from:
He, C., Zhu, Y., Guo, Y., et al. (2025). Rainfall variability and under-five 
child mortality in 59 low- and middle-income countries. Nature Water, 3, 881-889.

RSD Formula:
RSD = Σ[(Monthly_rainfall - Historical_mean) / Historical_std] × (Historical_mean / Annual_mean)

Author: Cheng He
"""

import pandas as pd
import numpy as np
from multiprocessing import Pool
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================
BASE_DIR = Path('.../DATA')
INPUT_DIR = BASE_DIR / 'pf_result'
OUTPUT_DIR = BASE_DIR / 'pf_rsd_results'
CITY_LIST_DIR = BASE_DIR / 'city_list'
PRECIP_DIR = Path('/..../precip_csv')
HISTORICAL_STATS_FILE = BASE_DIR / 'all_points_dhs_prep.csv'

MIN_YEAR = 1981
N_MONTHS = 12
N_PROCESSES = 50
HISTORICAL_PERIOD = (1980, 2020)  # Period for calculating historical statistics


# ============================================================================
# Historical Statistics Calculation
# ============================================================================
def calculate_historical_stats(precip_data):
    """
    Calculate historical statistics for RSD computation.
    
    Parameters
    ----------
    precip_data : pd.DataFrame
        Daily precipitation data with 'time' and 'tp' columns
    
    Returns
    -------
    dict
        Dictionary containing:
        - monthly_mean: Mean precipitation for each month (1-12)
        - monthly_std: Standard deviation of monthly totals
        - annual_mean: Mean annual precipitation
    """
    precip_data['time'] = pd.to_datetime(precip_data['time'])
    precip_data['year'] = precip_data['time'].dt.year
    precip_data['month'] = precip_data['time'].dt.month
    
    # Filter historical period
    hist_data = precip_data[
        (precip_data['year'] >= HISTORICAL_PERIOD[0]) & 
        (precip_data['year'] <= HISTORICAL_PERIOD[1])
    ]
    
    # Calculate monthly totals
    monthly_totals = hist_data.groupby(['year', 'month'])['tp'].sum().reset_index()
    
    # Calculate mean for each calendar month (1-12)
    monthly_mean = {}
    for month in range(1, 13):
        month_data = monthly_totals[monthly_totals['month'] == month]['tp']
        monthly_mean[month] = month_data.mean()
    
    # Calculate standard deviation of monthly totals across all months
    monthly_std = monthly_totals['tp'].std()
    
    # Calculate mean annual precipitation
    annual_totals = hist_data.groupby('year')['tp'].sum()
    annual_mean = annual_totals.mean()
    
    return {
        'monthly_mean': monthly_mean,
        'monthly_std': monthly_std,
        'annual_mean': annual_mean
    }


# ============================================================================
# RSD Calculation
# ============================================================================
def calc_monthly_rsd(target_date, precip_data, hist_stats):
    """
    Calculate standardized rainfall deviation for one month.
    
    Parameters
    ----------
    target_date : pd.Timestamp
        Target month date
    precip_data : pd.DataFrame
        Daily precipitation data
    hist_stats : dict
        Historical statistics from calculate_historical_stats()
    
    Returns
    -------
    float
        Standardized rainfall deviation for the month
    """
    year, month = target_date.year, target_date.month
    
    # Get actual monthly total
    monthly_data = precip_data[
        (precip_data['time'].dt.year == year) & 
        (precip_data['time'].dt.month == month)
    ]
    actual_total = monthly_data['tp'].sum()
    
    # Get historical statistics
    hist_mean = hist_stats['monthly_mean'][month]
    hist_std = hist_stats['monthly_std']
    annual_mean = hist_stats['annual_mean']
    
    # Calculate RSD for this month
    if hist_std == 0 or annual_mean == 0:
        return 0.0
    
    rsd = ((actual_total - hist_mean) / hist_std) * (hist_mean / annual_mean)
    
    return rsd


def calc_cumulative_rsd(target_date, precip_data, hist_stats, n_months):
    """
    Calculate cumulative RSD over multiple months.
    
    Parameters
    ----------
    target_date : pd.Timestamp
        End date (death date)
    precip_data : pd.DataFrame
        Daily precipitation data
    hist_stats : dict
        Historical statistics
    n_months : int
        Number of months to accumulate
    
    Returns
    -------
    float
        Cumulative RSD over n_months period
    """
    cumulative_rsd = 0.0
    
    for i in range(1, n_months + 1):
        month_date = target_date - pd.DateOffset(months=i)
        monthly_rsd = calc_monthly_rsd(month_date, precip_data, hist_stats)
        cumulative_rsd += monthly_rsd
    
    return cumulative_rsd


# ============================================================================
# Data Loading Functions
# ============================================================================
def load_precip_data(dhsid, survey_id):
    """Load precipitation data for a location"""
    survey_clean = survey_id.replace("Cote_d'Ivoire", "Cote_dIvoire")
    
    # Get city ID
    city_map = pd.read_csv(CITY_LIST_DIR / f'{survey_id}.csv')
    city_id = str(int(city_map[city_map['DHSID'] == dhsid]['city_id'].values[0]))
    
    # Load precipitation data
    precip = pd.read_csv(PRECIP_DIR / survey_clean / f'{city_id}.csv')
    precip['time'] = pd.to_datetime(precip['time'])
    
    return precip


def load_precomputed_stats(dhsid, stats_file):
    """
    Load precomputed historical statistics if available.
    
    Parameters
    ----------
    dhsid : str
        DHS cluster ID
    stats_file : Path
        Path to precomputed statistics file
    
    Returns
    -------
    dict or None
        Historical statistics or None if not available
    """
    try:
        stats_df = pd.read_csv(stats_file)
        location_stats = stats_df[stats_df['DHSID'] == dhsid]
        
        if len(location_stats) == 0:
            return None
        
        # Extract monthly means
        monthly_mean = {}
        for month in range(1, 13):
            col_name = f'{month}_average'
            if col_name in location_stats.columns:
                monthly_mean[month] = float(location_stats[col_name].values[0])
        
        # Extract standard deviation and annual mean
        hist_stats = {
            'monthly_mean': monthly_mean,
            'monthly_std': float(location_stats['month_std'].values[0]),
            'annual_mean': float(location_stats['all_average'].values[0])
        }
        
        return hist_stats
        
    except Exception as e:
        print(f"Warning: Could not load precomputed stats for {dhsid}: {e}")
        return None


# ============================================================================
# Main Processing Function
# ============================================================================
def process_file(filename):
    """Process single DHS file"""
    try:
        # Load data
        df = pd.read_csv(INPUT_DIR / filename)
        
        # Prepare dates
        df['death_date_lag_0'] = pd.to_datetime(df['death_date_lag_0'])
        df = df[df['death_date_lag_0'].dt.year >= MIN_YEAR].reset_index(drop=True)
        
        # Initialize result columns for monthly RSD
        for m in range(1, N_MONTHS + 1):
            df[f'rsd_month_{m}'] = 0.0
        
        # Initialize cumulative RSD columns
        for m in range(1, N_MONTHS + 1):
            df[f'rsd_cumulative_{m}'] = 0.0
        
        # Process each row
        for idx in range(len(df)):
            dhsid = df['DHSID'].iloc[idx]
            survey = df['DHS_survey'].iloc[idx]
            base_date = df['death_date_lag_0'].iloc[idx]
            
            # Load precipitation data
            precip = load_precip_data(dhsid, survey)
            
            # Try to load precomputed statistics, otherwise calculate
            hist_stats = load_precomputed_stats(dhsid, HISTORICAL_STATS_FILE)
            if hist_stats is None:
                hist_stats = calculate_historical_stats(precip)
            
            # Calculate RSD for each month
            monthly_rsds = []
            for m in range(1, N_MONTHS + 1):
                target_date = base_date - pd.DateOffset(months=m)
                rsd = calc_monthly_rsd(target_date, precip, hist_stats)
                df.at[idx, f'rsd_month_{m}'] = rsd
                monthly_rsds.append(rsd)
            
            # Calculate cumulative RSD
            for m in range(1, N_MONTHS + 1):
                cumulative = sum(monthly_rsds[:m])
                df.at[idx, f'rsd_cumulative_{m}'] = cumulative
            
            # Progress indicator
            if (idx + 1) % 100 == 0:
                print(f'{filename}: {idx + 1}/{len(df)} rows')
        
        # Calculate 12-month cumulative RSD (annual)
        df['rsd_annual'] = df['rsd_cumulative_12']
        
        # Separate positive and negative deviations
        df['rsd_positive'] = df['rsd_annual'].apply(lambda x: x if x > 0 else 0)
        df['rsd_negative'] = df['rsd_annual'].apply(lambda x: x if x < 0 else 0)
        
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
    print(f'Calculating RSD for historical period: {HISTORICAL_PERIOD[0]}-{HISTORICAL_PERIOD[1]}\n')
    
    with Pool(N_PROCESSES) as pool:
        results = pool.map(process_file, [f.name for f in files])
    
    success = sum(1 for r in results if r)
    print(f'\nCompleted: {success}/{len(files)} files')


if __name__ == '__main__':
    main()
