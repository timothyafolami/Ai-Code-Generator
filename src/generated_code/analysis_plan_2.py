import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import warnings
import json
from datetime import datetime
import statsmodels.api as sm
from functools import partial

def analyze_data(df):
    warnings.filterwarnings('ignore')
    
    # Make a copy to avoid modifying the original
    data = df.copy()
    
    # 1. Data Preparation
    # Check for missing values
    missing_values = data.isnull().sum()
    
    # Handle date column - create acquisition month cohorts
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
        data['acquisition_month'] = data['date'].dt.to_period('M').astype(str)
    
    # Create RFM-lite proxies
    if 'units_sold' in data.columns and 'revenue' in data.columns:
        data['avg_price_per_unit'] = data['revenue'] / data['units_sold']
        data['total_value'] = data['revenue'] * (1 - data['returned'])
    
    # Handle missing region explicitly
    if 'region' in data.columns:
        data['region'] = data['region'].fillna('Unknown')
    
    # 2. Descriptive Cohort Analysis
    
    # Retention rate by cohort
    cohort_retention = data.groupby('acquisition_month')['repeat_customer'].mean().reset_index()
    cohort_retention.columns = ['acquisition_month', 'retention_rate']
    
    # Retention by product
    product_retention = data.groupby('product')['repeat_customer'].agg(['mean', 'count']).reset_index()
    product_retention.columns = ['product', 'retention_rate', 'customer_count']
    
    # Retention by channel
    channel_retention = data.groupby('channel')['repeat_customer'].agg(['mean', 'count']).reset_index()
    channel_retention.columns = ['channel', 'retention_rate', 'customer_count']
    
    # Retention by region
    region_retention = data.groupby('region')['repeat_customer'].agg(['mean', 'count']).reset_index()
    region_retention.columns = ['region', 'retention_rate', 'customer_count']
    
    # Return rate analysis
    return_by_product = data.groupby('product')['returned'].mean().reset_index()
    return_by_product.columns = ['product', 'return_rate']
    
    return_by_channel = data.groupby('channel')['returned'].mean().reset_index()
    return_by_channel.columns = ['channel', 'return_rate']
    
    # 3. Predictive Models
    
    # Define features for models
    features = ['product', 'region', 'channel', 'customer_age', 'unit_price', 
                'discount_rate', 'units_sold', 'marketing_spend', 'revenue']
    
    # Filter only features that exist in the dataframe
    features = [f for f in features if f in data.columns]
    
    # Separate categorical and numerical features
    categorical_features = [f for f in features if data[f].dtype == 'object']
    numerical_features = [f for f in features if data[f].dtype != 'object' and f in features]
    
    # Prepare preprocessing pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Model results storage
    model_results = {}
    
    # Function to train and evaluate a model
    def train_evaluate_model(X, y, model_name, model):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Create and train the pipeline
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
        clf.fit(X_train, y_train)
        
        # Predictions
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)
        
        # Metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        precision_at_10 = precision_score(y_test, (y_pred_proba >= np.percentile(y_pred_proba, 90)).astype(int))
        conf_matrix = confusion_matrix(y_test, y_pred).tolist()
        
        # Feature importance (for logistic regression)
        feature_importance = {}
        if model_name == 'logistic_regression':
            # Get feature names after one-hot encoding
            cat_features = clf.named_steps['preprocessor'].transformers_[1][1]['onehot'].get_feature_names_out(categorical_features)
            all_features = np.concatenate([numerical_features, cat_features])
            
            # Get coefficients
            coefficients = clf.named_steps['classifier'].coef_[0]
            for feat, coef in zip(all_features, coefficients):
                feature_importance[str(feat)] = float(coef)
        
        # For gradient boosting
        elif model_name == 'gradient_boosting':
            feature_importance = dict(zip(features, clf.named_steps['classifier'].feature_importances_))
        
        return {
            'model': clf,
            'auc_score': auc_score,
            'precision_at_10': precision_at_10,
            'confusion_matrix': conf_matrix,
            'feature_importance': feature_importance
        }
    
    # Train models for repeat_customer prediction
    if 'repeat_customer' in data.columns:
        X = data[features]
        y = data['repeat_customer']
        
        # Logistic Regression
        model_results['repeat_customer_logistic'] = train_evaluate_model(
            X, y, 'logistic_regression', LogisticRegression(max_iter=1000, class_weight='balanced')
        )
        
        # Gradient Boosting
        model_results['repeat_customer_gb'] = train_evaluate_model(
            X, y, 'gradient_boosting', GradientBoostingClassifier(random_state=42)
        )
    
    # Train models for returned prediction
    if 'returned' in data.columns:
        X = data[features]
        y = data['returned']
        
        # Logistic Regression
        model_results['returned_logistic'] = train_evaluate_model(
            X, y, 'logistic_regression', LogisticRegression(max_iter=1000, class_weight='balanced')
        )
        
        # Gradient Boosting
        model_results['returned_gb'] = train_evaluate_model(
            X, y, 'gradient_boosting', GradientBoostingClassifier(random_state=42)
        )
    
    # 4. Identify High-Value Customer Segments
    
    # Create a customer value score (simplified LTV proxy)
    data['customer_value'] = data['revenue'] * (1 - data['returned']) * (1 + data['repeat_customer'])
    
    # Identify top segments
    segment_columns = ['product', 'channel', 'region']
    segment_columns = [col for col in segment_columns if col in data.columns]
    
    if segment_columns:
        high_value_segments = data.groupby(segment_columns).agg({
            'customer_value': 'mean',
            'repeat_customer': 'mean',
            'returned': 'mean',
            'customer_age': 'mean' if 'customer_age' in data.columns else 'count',
            'revenue': 'sum'
        }).reset_index()
        
        high_value_segments = high_value_segments.sort_values('customer_value', ascending=False).head(5)
    else:
        high_value_segments = pd.DataFrame()
    
    # 5. Generate Action Rules
    
    # Simple rules based on feature importance
    action_rules = []
    
    # For repeat customers
    if 'repeat_customer_logistic' in model_results:
        top_features = sorted(model_results['repeat_customer_logistic']['feature_importance'].items(), 
                             key=lambda x: abs(x[1]), reverse=True)[:5]
        
        action_rules.append({
            'target': 'increase_retention',
            'top_factors': [{'feature': f[0], 'importance': f[1]} for f in top_features]
        })
    
    # For returns
    if 'returned_logistic' in model_results:
        top_features = sorted(model_results['returned_logistic']['feature_importance'].items(), 
                             key=lambda x: abs(x[1]), reverse=True)[:5]
        
        action_rules.append({
            'target': 'reduce_returns',
            'top_factors': [{'feature': f[0], 'importance': f[1]} for f in top_features]
        })
    
    # 6. Prepare Results
    
    results = {
        'cohort_analysis': {
            'cohort_retention': cohort_retention.to_dict('records'),
            'product_retention': product_retention.to_dict('records'),
            'channel_retention': channel_retention.to_dict('records'),
            'region_retention': region_retention.to_dict('records')
        },
        'return_analysis': {
            'return_by_product': return_by_product.to_dict('records'),
            'return_by_channel': return_by_channel.to_dict('records')
        },
        'predictive_models': {
            'repeat_customer': {
                'logistic_regression': {
                    'auc_score': model_results.get('repeat_customer_logistic', {}).get('auc_score'),
                    'precision_at_10': model_results.get('repeat_customer_logistic', {}).get('precision_at_10'),
                    'top_features': sorted(model_results.get('repeat_customer_logistic', {}).get('feature_importance', {}).items(), 
                                          key=lambda x: abs(x[1]), reverse=True)[:10]
                },
                'gradient_boosting': {
                    'auc_score': model_results.get('repeat_customer_gb', {}).get('auc_score'),
                    'precision_at_10': model_results.get('repeat_customer_gb', {}).get('precision_at_10'),
                    'top_features': sorted(model_results.get('repeat_customer_gb', {}).get('feature_importance', {}).items(), 
                                          key=lambda x: abs(x[1]), reverse=True)[:10]
                }
            },
            'returned': {
                'logistic_regression': {
                    'auc_score': model_results.get('returned_logistic', {}).get('auc_score'),
                    'precision_at_10': model_results.get('returned_logistic', {}).get('precision_at_10'),
                    'top_features': sorted(model_results.get('returned_logistic', {}).get('feature_importance', {}).items(), 
                                          key=lambda x: abs(x[1]), reverse=True)[:10]
                },
                'gradient_boosting': {
                    'auc_score': model_results.get('returned_gb', {}).get('auc_score'),
                    'precision_at_10': model_results.get('returned_gb', {}).get('precision_at_10'),
                    'top_features': sorted(model_results.get('returned_gb', {}).get('feature_importance', {}).items(), 
                                          key=lambda x: abs(x[1]), reverse=True)[:10]
                }
            }
        },
        'high_value_segments': high_value_segments.to_dict('records'),
        'action_rules': action_rules,
        'data_summary': {
            'row_count': len(data),
            'repeat_customer_rate': data['repeat_customer'].mean() if 'repeat_customer' in data.columns else None,
            'return_rate': data['returned'].mean() if 'returned' in data.columns else None,
            'missing_values': missing_values.to_dict()
        }
    }
    
    # Convert any numpy types to Python native types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        else:
            return obj
    
    # Recursively convert all values in the results dictionary
    def convert_dict_values(d):
        if isinstance(d, dict):
            return {k: convert_dict_values(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [convert_dict_values(v) for v in d]
        else:
            return convert_to_serializable(d)
    
    results = convert_dict_values(results)
    
    return results
