import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from datetime import datetime
import json

def analyze_data(df):
    # Suppress warnings
    warnings.filterwarnings('ignore')
    
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Data preparation and cleaning
    # Handle missing values in region
    data['region_missing'] = data['region'].isna().astype(int)
    data['region'] = data['region'].fillna('MISSING')
    
    # Convert date to time features if date column exists
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        data['month'] = data['date'].dt.month
        data['quarter'] = data['date'].dt.quarter
        data['year'] = data['date'].dt.year
        data['day_of_week'] = data['date'].dt.dayofweek
    
    # Calculate revenue if not already present
    if 'revenue' not in data.columns and 'units_sold' in data.columns and 'unit_price' in data.columns:
        data['revenue'] = data['units_sold'] * data['unit_price'] * (1 - data['discount_rate'])
    
    # Create interaction terms
    data['price_discount_interaction'] = data['unit_price'] * data['discount_rate']
    
    # Create marketing spend per unit
    if 'marketing_spend' in data.columns and 'units_sold' in data.columns:
        data['marketing_per_unit'] = data['marketing_spend'] / data['units_sold'].replace(0, 1)
    
    # Exploratory Data Analysis
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in data.columns if col not in numeric_cols and data[col].nunique() < 30]
    
    # Correlation analysis of numeric variables
    corr_matrix = data[numeric_cols].corr()
    
    # Prepare data for modeling
    # Define features and target
    target = 'revenue'
    features = [col for col in data.columns if col != target and col not in ['date']]
    
    # Split data for training and testing
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Identify categorical and numerical features
    cat_features = [col for col in features if col in categorical_cols]
    num_features = [col for col in features if col in numeric_cols and col != target]
    
    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])
    
    # Elastic Net model with cross-validation for hyperparameter tuning
    elastic_net = Pipeline([
        ('preprocessor', preprocessor),
        ('model', ElasticNet(random_state=42))
    ])
    
    # Parameter grid for ElasticNet
    param_grid = {
        'model__alpha': [0.001, 0.01, 0.1, 1, 10],
        'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    }
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(elastic_net, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    
    # Predictions
    y_pred = best_model.predict(X_test)
    
    # Model evaluation
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Statistical modeling for elasticity estimation
    # Prepare data for statsmodels (log transformation for elasticity)
    model_data = data.copy()
    
    # Log transform for elasticity calculation
    for col in ['revenue', 'unit_price', 'discount_rate', 'marketing_spend']:
        if col in model_data.columns:
            model_data[f'log_{col}'] = np.log(model_data[col].replace(0, 0.01))
    
    # Formula for statsmodels (adjust based on available columns)
    formula_parts = ["log_revenue ~ log_unit_price + log_discount_rate"]
    
    if 'marketing_spend' in model_data.columns:
        formula_parts[0] += " + log_marketing_spend"
    
    for cat in categorical_cols:
        formula_parts.append(f"C({cat})")
    
    formula = formula_parts[0]
    if len(formula_parts) > 1:
        formula += " + " + " + ".join(formula_parts[1:])
    
    # Fit the model
    try:
        ols_model = smf.ols(formula=formula, data=model_data).fit()
        ols_summary = ols_model.summary()
        
        # Extract coefficients for elasticity
        price_elasticity = ols_model.params.get('log_unit_price', 0)
        discount_elasticity = ols_model.params.get('log_discount_rate', 0)
        marketing_elasticity = ols_model.params.get('log_marketing_spend', 0)
        
    except Exception as e:
        # Fallback to simpler model if the full model fails
        simple_formula = "log_revenue ~ log_unit_price + log_discount_rate"
        ols_model = smf.ols(formula=simple_formula, data=model_data).fit()
        ols_summary = ols_model.summary()
        
        # Extract coefficients for elasticity
        price_elasticity = ols_model.params.get('log_unit_price', 0)
        discount_elasticity = ols_model.params.get('log_discount_rate', 0)
        marketing_elasticity = 0
    
    # Calculate elasticities by product if product column exists
    product_elasticities = {}
    if 'product' in data.columns:
        for product in data['product'].unique():
            product_data = model_data[model_data['product'] == product]
            if len(product_data) > 30:  # Only calculate if enough data
                try:
                    product_model = smf.ols(formula=simple_formula, data=product_data).fit()
                    product_elasticities[product] = {
                        'price_elasticity': product_model.params.get('log_unit_price', 0),
                        'discount_elasticity': product_model.params.get('log_discount_rate', 0)
                    }
                except:
                    product_elasticities[product] = {
                        'price_elasticity': np.nan,
                        'discount_elasticity': np.nan
                    }
    
    # Calculate marketing ROI
    marketing_roi = {}
    if 'marketing_spend' in data.columns and 'revenue' in data.columns:
        # Group by product and channel if available
        group_cols = [col for col in ['product', 'channel'] if col in data.columns]
        
        if group_cols:
            grouped = data.groupby(group_cols).agg({
                'marketing_spend': 'sum',
                'revenue': 'sum'
            }).reset_index()
            
            for _, row in grouped.iterrows():
                group_key = tuple(row[col] for col in group_cols)
                marketing_roi[str(group_key)] = {
                    'marketing_spend': row['marketing_spend'],
                    'revenue': row['revenue'],
                    'roi': row['revenue'] / row['marketing_spend'] if row['marketing_spend'] > 0 else np.nan
                }
        else:
            total_marketing = data['marketing_spend'].sum()
            total_revenue = data['revenue'].sum()
            marketing_roi['overall'] = {
                'marketing_spend': total_marketing,
                'revenue': total_revenue,
                'roi': total_revenue / total_marketing if total_marketing > 0 else np.nan
            }
    
    # Analyze interaction effects
    interaction_effects = {}
    if 'discount_rate' in data.columns and 'units_sold' in data.columns and 'revenue' in data.columns:
        # Group by discount rate ranges
        data['discount_bin'] = pd.qcut(data['discount_rate'], 5, duplicates='drop')
        discount_analysis = data.groupby('discount_bin').agg({
            'units_sold': 'mean',
            'revenue': 'mean'
        }).reset_index()
        
        interaction_effects['discount_impact'] = discount_analysis.to_dict('records')
    
    # Generate recommendations
    recommendations = []
    
    # Price recommendations based on elasticity
    if abs(price_elasticity) > 1:  # Elastic demand
        if price_elasticity < 0:  # Negative elasticity (normal goods)
            recommendations.append({
                'action': 'Decrease prices',
                'reason': 'Price elasticity > 1 indicates elastic demand; lowering prices should increase revenue',
                'elasticity': price_elasticity
            })
        else:  # Positive elasticity (luxury/Veblen goods)
            recommendations.append({
                'action': 'Increase prices',
                'reason': 'Positive price elasticity indicates luxury/Veblen goods; increasing prices may increase revenue',
                'elasticity': price_elasticity
            })
    else:  # Inelastic demand
        recommendations.append({
            'action': 'Increase prices',
            'reason': 'Price elasticity < 1 indicates inelastic demand; increasing prices should increase revenue',
            'elasticity': price_elasticity
        })
    
    # Discount recommendations
    if discount_elasticity < -1:
        recommendations.append({
            'action': 'Increase discounts',
            'reason': 'Discount elasticity < -1 indicates discounts effectively drive revenue',
            'elasticity': discount_elasticity
        })
    else:
        recommendations.append({
            'action': 'Reduce discounts',
            'reason': 'Discount elasticity > -1 indicates discounts are not effectively driving revenue',
            'elasticity': discount_elasticity
        })
    
    # Marketing recommendations
    if marketing_elasticity > 0:
        recommendations.append({
            'action': 'Increase marketing spend',
            'reason': 'Positive marketing elasticity indicates effective marketing',
            'elasticity': marketing_elasticity
        })
    else:
        recommendations.append({
            'action': 'Optimize marketing spend',
            'reason': 'Low marketing elasticity indicates need for better targeting',
            'elasticity': marketing_elasticity
        })
    
    # Compile results
    results = {
        'model_performance': {
            'r2': r2,
            'rmse': rmse,
            'best_params': grid_search.best_params_
        },
        'elasticity_estimates': {
            'price_elasticity': price_elasticity,
            'discount_elasticity': discount_elasticity,
            'marketing_elasticity': marketing_elasticity
        },
        'product_elasticities': product_elasticities,
        'marketing_roi': marketing_roi,
        'interaction_effects': interaction_effects,
        'recommendations': recommendations
    }
    
    return results
