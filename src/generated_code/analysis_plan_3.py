import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
import warnings
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
from typing import Dict, Any, List, Tuple

def analyze_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform comprehensive channel and regional performance benchmarking with missingness sensitivity analysis.
    
    Args:
        df: DataFrame with sales, marketing, and regional data
    
    Returns:
        Dictionary containing analysis results
    """
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Ensure date column is datetime if it exists
    date_cols = [col for col in data.columns if 'date' in col.lower()]
    if date_cols:
        data[date_cols[0]] = pd.to_datetime(data[date_cols[0]])
    
    # Basic data validation and cleaning
    result = {
        "summary": {},
        "channel_analysis": {},
        "region_analysis": {},
        "missingness_analysis": {},
        "time_trends": {},
        "recommendations": {}
    }
    
    # Check for required columns
    required_cols = ['channel', 'region', 'revenue', 'units_sold', 'unit_price', 
                     'marketing_spend', 'return_rate']
    
    # Adapt to available columns
    available_cols = [col for col in required_cols if col in data.columns]
    missing_cols = [col for col in required_cols if col not in data.columns]
    
    if missing_cols:
        result["summary"]["missing_columns"] = missing_cols
        # Try to derive missing columns if possible
        if 'revenue' not in data.columns and 'units_sold' in data.columns and 'unit_price' in data.columns:
            data['revenue'] = data['units_sold'] * data['unit_price']
            available_cols.append('revenue')
        if 'unit_price' not in data.columns and 'revenue' in data.columns and 'units_sold' in data.columns:
            # avoid division by zero
            data['unit_price'] = data['revenue'] / data['units_sold'].replace({0: np.nan})
            available_cols.append('unit_price')
    
    # Basic dataset summary
    result["summary"]["row_count"] = len(data)
    result["summary"]["available_columns"] = available_cols
    
    # Check for channel and region columns
    if 'channel' not in data.columns or 'region' not in data.columns:
        result["summary"]["error"] = "Required columns 'channel' or 'region' missing"
        return result
    
    # Analyze missingness in region column
    region_missing = int(data['region'].isna().sum())
    result["missingness_analysis"]["region_missing_count"] = region_missing
    result["missingness_analysis"]["region_missing_percent"] = round(region_missing / len(data) * 100, 2)
    
    # Create a flag for missing region
    data['region_missing'] = data['region'].isna()
    
    # 1. Aggregation: compute KPIs by channel and region
    
    # Build aggregation map dynamically based on available columns
    agg_map_channel = {}
    if 'revenue' in data.columns:
        agg_map_channel['revenue'] = 'sum'
    if 'units_sold' in data.columns:
        agg_map_channel['units_sold'] = 'sum'
    if 'unit_price' in data.columns:
        agg_map_channel['unit_price'] = 'mean'
    if 'return_rate' in data.columns:
        agg_map_channel['return_rate'] = 'mean'
    if 'marketing_spend' in data.columns:
        agg_map_channel['marketing_spend'] = 'sum'
    
    # Channel KPIs
    if agg_map_channel:
        channel_kpis = data.groupby('channel').agg(agg_map_channel).reset_index()
    else:
        channel_kpis = pd.DataFrame(columns=['channel'])
    
    # Calculate marketing efficiency
    if 'revenue' in channel_kpis.columns and 'marketing_spend' in channel_kpis.columns:
        channel_kpis['marketing_efficiency'] = channel_kpis['revenue'] / channel_kpis['marketing_spend'].replace({0: np.nan})
    else:
        channel_kpis['marketing_efficiency'] = np.nan
    
    if 'revenue' in channel_kpis.columns and 'units_sold' in channel_kpis.columns:
        channel_kpis['avg_revenue_per_transaction'] = channel_kpis['revenue'] / channel_kpis['units_sold'].replace({0: np.nan})
    else:
        channel_kpis['avg_revenue_per_transaction'] = np.nan
    
    # Region KPIs (excluding missing regions)
    region_data = data[~data['region_missing']]
    agg_map_region = {}
    if 'revenue' in region_data.columns:
        agg_map_region['revenue'] = 'sum'
    if 'units_sold' in region_data.columns:
        agg_map_region['units_sold'] = 'sum'
    if 'unit_price' in region_data.columns:
        agg_map_region['unit_price'] = 'mean'
    if 'return_rate' in region_data.columns:
        agg_map_region['return_rate'] = 'mean'
    if 'marketing_spend' in region_data.columns:
        agg_map_region['marketing_spend'] = 'sum'
    
    if agg_map_region:
        region_kpis = region_data.groupby('region').agg(agg_map_region).reset_index()
    else:
        region_kpis = pd.DataFrame(columns=['region'])
    
    # Calculate marketing efficiency for regions
    if 'revenue' in region_kpis.columns and 'marketing_spend' in region_kpis.columns:
        region_kpis['marketing_efficiency'] = region_kpis['revenue'] / region_kpis['marketing_spend'].replace({0: np.nan})
    else:
        region_kpis['marketing_efficiency'] = np.nan
    
    if 'revenue' in region_kpis.columns and 'units_sold' in region_kpis.columns:
        region_kpis['avg_revenue_per_transaction'] = region_kpis['revenue'] / region_kpis['units_sold'].replace({0: np.nan})
    else:
        region_kpis['avg_revenue_per_transaction'] = np.nan
    
    # Store aggregated KPIs in result
    result["channel_analysis"]["kpis"] = channel_kpis.to_dict(orient='records')
    result["region_analysis"]["kpis"] = region_kpis.to_dict(orient='records')
    
    # 2. Statistical comparison
    
    # ANOVA for continuous KPIs by channel
    kpi_metrics = ['revenue', 'unit_price', 'marketing_efficiency', 'avg_revenue_per_transaction']
    channel_stats = {}
    
    for metric in kpi_metrics:
        if metric in data.columns or metric in channel_kpis.columns:
            try:
                # For metrics that are in the original data
                if metric in data.columns:
                    groups = [data[data['channel'] == channel][metric].dropna() 
                              for channel in data['channel'].unique()]
                    # Ensure at least two groups with data
                    if sum([1 for g in groups if len(g) > 0]) < 2:
                        raise ValueError("Not enough groups with data for ANOVA")
                    f_stat, p_val = stats.f_oneway(*groups)
                    
                    # Perform Tukey's HSD test for pairwise comparisons
                    tukey_data = pd.DataFrame({
                        'score': data[metric].dropna(),
                        'group': data['channel'][data[metric].notna()]
                    })
                    # only perform if more than one group present
                    if tukey_data['group'].nunique() > 1:
                        tukey = pairwise_tukeyhsd(tukey_data['score'], tukey_data['group'])
                        pairs = []
                        for i, row in enumerate(tukey.summary().data[1:]):
                            pairs.append({
                                'group1': row[0],
                                'group2': row[1],
                                'meandiff': float(row[2]),
                                'p_adj': float(row[3]),
                                'significant': True if str(row[4]).lower() in ('true', '1') else False
                            })
                    else:
                        pairs = []
                    
                    channel_stats[metric] = {
                        'f_statistic': float(f_stat),
                        'p_value': float(p_val),
                        'significant': p_val < 0.05,
                        'pairwise_comparisons': pairs
                    }
            except Exception:
                channel_stats[metric] = {'error': 'Could not perform statistical test'}
    
    result["channel_analysis"]["statistical_tests"] = channel_stats
    
    # Chi-square test for return rates by channel
    if 'return_rate' in data.columns:
        try:
            # Create contingency table for return rates
            # We'll categorize return rates as high/low based on median
            median_return = data['return_rate'].median()
            contingency = pd.crosstab(
                data['channel'], 
                data['return_rate'] > median_return
            )
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            
            result["channel_analysis"]["return_rate_test"] = {
                'chi2': float(chi2),
                'p_value': float(p),
                'dof': int(dof),
                'significant': p < 0.05
            }
        except Exception:
            result["channel_analysis"]["return_rate_test"] = {'error': 'Could not perform chi-square test'}
    
    # 3. Time trends analysis
    if date_cols:
        date_col = date_cols[0]
        
        # Create 4-week rolling aggregates by channel
        data = data.sort_values(by=date_col)
        
        # Set date as index for resampling
        data_dated = data.set_index(date_col)
        
        # Get weekly aggregates by channel
        channels = data['channel'].unique()
        time_trends = {}
        
        for channel in channels:
            channel_data = data_dated[data_dated['channel'] == channel]
            
            # Skip if not enough data
            if len(channel_data) < 7:
                continue
                
            try:
                # Weekly revenue
                if 'revenue' in channel_data.columns:
                    weekly_revenue = channel_data['revenue'].resample('W').sum()
                    
                    # Calculate 4-week rolling average
                    rolling_revenue = weekly_revenue.rolling(window=4, min_periods=1).mean()
                    
                    # Calculate growth rate (week-over-week)
                    growth_rate = weekly_revenue.pct_change().fillna(0)
                    
                    # Store results
                    time_trends[channel] = {
                        'weekly_revenue': weekly_revenue.to_dict(),
                        'rolling_4week_avg': rolling_revenue.to_dict(),
                        'wow_growth_rate': growth_rate.to_dict()
                    }
                else:
                    time_trends[channel] = {'error': 'No revenue column for time trends'}
            except Exception:
                time_trends[channel] = {'error': 'Could not calculate time trends'}
        
        result["time_trends"]["by_channel"] = time_trends
    
    # 4. Missingness diagnostic
    
    # Check if missing regions are associated with specific channels
    region_missing_by_channel = data.groupby('channel')['region_missing'].mean().to_dict()
    result["missingness_analysis"]["region_missing_by_channel"] = region_missing_by_channel
    
    # Check if missing regions have different KPIs
    agg_map_missing = {}
    for col in ['revenue', 'units_sold', 'unit_price', 'return_rate', 'marketing_spend']:
        if col in data.columns:
            agg_map_missing[col] = 'mean'
    
    if agg_map_missing:
        missing_vs_present = data.groupby('region_missing').agg(agg_map_missing).to_dict()
    else:
        missing_vs_present = {}
    
    result["missingness_analysis"]["kpi_comparison"] = missing_vs_present
    
    # Statistical test for differences in KPIs between missing and non-missing regions
    missingness_tests = {}
    
    for metric in ['revenue', 'units_sold', 'unit_price', 'return_rate', 'marketing_spend']:
        if metric in data.columns:
            try:
                group_missing = data[data['region_missing']][metric].dropna()
                group_present = data[~data['region_missing']][metric].dropna()
                # require at least one observation in each group
                if len(group_missing) < 2 or len(group_present) < 2:
                    raise ValueError("Not enough data for t-test")
                t_stat, p_val = stats.ttest_ind(
                    group_missing,
                    group_present,
                    equal_var=False  # Welch's t-test
                )
                
                missingness_tests[metric] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_val),
                    'significant': p_val < 0.05
                }
            except Exception:
                missingness_tests[metric] = {'error': 'Could not perform t-test'}
    
    result["missingness_analysis"]["statistical_tests"] = missingness_tests
    
    # 5. Sensitivity analysis with region imputation
    
    # Only attempt imputation if we have enough data and missing regions
    if region_missing > 10 and region_missing < len(data) * 0.5:
        try:
            # Prepare data for imputation
            features = ['channel', 'revenue', 'units_sold', 'unit_price', 'marketing_spend', 'return_rate']
            available_features = [f for f in features if f in data.columns]
            
            # Create training data (non-missing regions)
            X_train = data[~data['region_missing']][available_features]
            y_train = data[~data['region_missing']]['region']
            
            # Handle categorical features
            X_train = pd.get_dummies(X_train, drop_first=True)
            
            # Train a simple classifier
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            clf.fit(X_train, y_train)
            
            # Prepare data for prediction
            X_missing = data[data['region_missing']][available_features]
            X_missing = pd.get_dummies(X_missing, drop_first=True)
            
            # Ensure X_missing has same columns as X_train
            for col in X_train.columns:
                if col not in X_missing.columns:
                    X_missing[col] = 0
            X_missing = X_missing[X_train.columns]
            
            # Predict regions
            predicted_regions = clf.predict(X_missing)
            
            # Create a copy of data with imputed regions
            imputed_data = data.copy()
            imputed_data.loc[imputed_data['region_missing'], 'region'] = predicted_regions
            
            # Recalculate KPIs with imputed regions
            agg_map_imputed = {}
            if 'revenue' in imputed_data.columns:
                agg_map_imputed['revenue'] = 'sum'
            if 'units_sold' in imputed_data.columns:
                agg_map_imputed['units_sold'] = 'sum'
            if 'unit_price' in imputed_data.columns:
                agg_map_imputed['unit_price'] = 'mean'
            if 'return_rate' in imputed_data.columns:
                agg_map_imputed['return_rate'] = 'mean'
            if 'marketing_spend' in imputed_data.columns:
                agg_map_imputed['marketing_spend'] = 'sum'
            
            if agg_map_imputed:
                imputed_region_kpis = imputed_data.groupby('region').agg(agg_map_imputed).reset_index()
            else:
                imputed_region_kpis = pd.DataFrame(columns=['region'])
            
            if 'revenue' in imputed_region_kpis.columns and 'marketing_spend' in imputed_region_kpis.columns:
                imputed_region_kpis['marketing_efficiency'] = imputed_region_kpis['revenue'] / imputed_region_kpis['marketing_spend'].replace({0: np.nan})
            else:
                imputed_region_kpis['marketing_efficiency'] = np.nan
            
            # Compare rankings before and after imputation
            if 'marketing_efficiency' in region_kpis.columns:
                original_ranks = region_kpis.sort_values('marketing_efficiency', ascending=False)['region'].tolist()
            else:
                original_ranks = []
            if 'marketing_efficiency' in imputed_region_kpis.columns:
                imputed_ranks = imputed_region_kpis.sort_values('marketing_efficiency', ascending=False)['region'].tolist()
            else:
                imputed_ranks = []
            
            result["missingness_analysis"]["imputation"] = {
                'original_region_ranks': original_ranks,
                'imputed_region_ranks': imputed_ranks,
                'rank_changed': original_ranks != imputed_ranks,
                'imputed_region_distribution': imputed_data.loc[data['region_missing'], 'region'].value_counts().to_dict()
            }
        except Exception:
            result["missingness_analysis"]["imputation"] = {'error': 'Could not perform imputation analysis'}
    
    # 6. Recommendations
    
    # Channel recommendations based on marketing efficiency
    channel_recommendations = channel_kpis.sort_values('marketing_efficiency', ascending=False).to_dict(orient='records')
    
    # Simple reallocation simulation
    if len(channel_kpis) > 1:
        try:
            # Find best and worst performing channels
            best_channel = channel_kpis.loc[channel_kpis['marketing_efficiency'].idxmax()]
            worst_channel = channel_kpis.loc[channel_kpis['marketing_efficiency'].idxmin()]
            
            # Simulate moving 20% of marketing spend from worst to best channel
            reallocation_amount = worst_channel['marketing_spend'] * 0.2
            
            # Calculate expected revenue impact
            expected_revenue_gain = reallocation_amount * best_channel['marketing_efficiency']
            expected_revenue_loss = reallocation_amount * worst_channel['marketing_efficiency']
            net_impact = expected_revenue_gain - expected_revenue_loss
            
            result["recommendations"]["marketing_reallocation"] = {
                'from_channel': worst_channel['channel'],
                'to_channel': best_channel['channel'],
                'reallocation_amount': float(reallocation_amount),
                'expected_revenue_gain': float(expected_revenue_gain),
                'expected_revenue_loss': float(expected_revenue_loss),
                'net_revenue_impact': float(net_impact)
            }
        except Exception:
            result["recommendations"]["marketing_reallocation"] = {'error': 'Could not calculate reallocation impact'}
    
    # Region recommendations
    region_recommendations = region_kpis.sort_values('marketing_efficiency', ascending=False).to_dict(orient='records')
    
    result["recommendations"]["channel_ranking"] = channel_recommendations
    result["recommendations"]["region_ranking"] = region_recommendations
    
    # Data quality recommendations
    if region_missing / len(data) > 0.1:  # If more than 10% missing
        result["recommendations"]["data_quality"] = {
            "issue": "High percentage of missing region data",
            "impact": "May bias channel performance metrics" if any(v.get('significant') for v in missingness_tests.values()) else "Limited impact on analysis",
            "action": "Implement region capture in sales system and consider imputation for historical data"
        }
    
    return result
