import os
import sys
import pandas as pd
import numpy as np
import warnings
import re
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("Optimized Product Pricing Model - Target SMAPE < 5%")
print("="*70)

try:
    test_gpu = xgb.XGBRegressor(tree_method='gpu_hist', gpu_id=0)
    USE_GPU = True
    print("GPU acceleration: Enabled")
except:
    USE_GPU = False
    print("GPU acceleration: Not available (using CPU)")
print("="*70 + "\n")

print("Loading dataset...")
train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')
print(f"Training samples: {len(train_df):,}")
print(f"Test samples: {len(test_df):,}\n")

print("Advanced feature extraction...")

def extract_features_advanced(df, is_train=True, encodings=None):
    """Advanced feature engineering for SMAPE < 5%"""
    data = df.copy()
    
    # Extract value with better handling
    data['value'] = data['catalog_content'].apply(
        lambda x: float(re.search(r'Value:\s*([\d.]+)', str(x)).group(1)) 
        if pd.notna(x) and re.search(r'Value:\s*([\d.]+)', str(x)) else np.nan
    )
    
    # Extract unit with normalization
    def extract_unit_normalized(text):
        if pd.isna(text):
            return 'unknown'
        match = re.search(r'Unit:\s*([^\n]+)', str(text))
        if match:
            unit = match.group(1).strip().lower()
            # Normalize common units
            unit_map = {
                'fluid ounce': 'fl_oz',
                'ounce': 'oz',
                'pound': 'lb',
                'count': 'count',
                'gram': 'g',
                'kilogram': 'kg',
                'milliliter': 'ml',
                'liter': 'l'
            }
            for key, val in unit_map.items():
                if key in unit:
                    return val
            return unit[:20]  # Truncate long units
        return 'unknown'
    
    data['unit'] = data['catalog_content'].apply(extract_unit_normalized)
    
    # Extract item name
    data['item_name'] = data['catalog_content'].apply(
        lambda x: re.search(r'Item Name:\s*([^\n]+)', str(x)).group(1).strip() 
        if pd.notna(x) and re.search(r'Item Name:\s*([^\n]+)', str(x)) else ''
    )
    
    # Extract pack quantity with more patterns
    def get_pack_quantity(text):
        patterns = [
            r'Pack of (\d+)', r'\(Pack of (\d+)\)', r'(\d+)[- ]Pack', 
            r'(\d+) ct', r'(\d+) Count', r'(\d+)-Count',
            r'(\d+) Pack', r'(\d+)pk'
        ]
        for pattern in patterns:
            match = re.search(pattern, str(text), re.I)
            if match:
                qty = int(match.group(1))
                return min(qty, 100)  # Cap at 100 to avoid outliers
        return 1
    
    data['pack_qty'] = data['catalog_content'].apply(get_pack_quantity)
    
    # Extract brand (more sophisticated)
    def extract_brand(name):
        name_str = str(name).strip()
        if len(name_str) == 0:
            return 'unknown'
        # Remove common prefixes
        name_str = re.sub(r'^(The|A|An)\s+', '', name_str, flags=re.I)
        # Get first word/brand
        words = name_str.split()
        if len(words) > 0:
            brand = words[0].lower()
            # Remove special chars
            brand = re.sub(r'[^a-z0-9]', '', brand)
            return brand if len(brand) > 0 else 'unknown'
        return 'unknown'
    
    data['brand'] = data['item_name'].apply(extract_brand)
    
    # Text features
    data['catalog_length'] = data['catalog_content'].fillna('').str.len()
    data['name_length'] = data['item_name'].fillna('').str.len()
    data['word_count'] = data['item_name'].fillna('').str.split().str.len()
    data['unique_words'] = data['item_name'].fillna('').apply(
        lambda x: len(set(str(x).lower().split()))
    )
    data['avg_word_length'] = data['name_length'] / (data['word_count'] + 1)
    data['unique_ratio'] = data['unique_words'] / (data['word_count'] + 1)
    
    # Bullet points and structure
    data['bullet_points'] = data['catalog_content'].fillna('').str.count('Bullet Point')
    data['special_chars'] = data['item_name'].fillna('').apply(
        lambda x: sum(not c.isalnum() and not c.isspace() for c in str(x))
    )
    data['digit_count'] = data['item_name'].fillna('').str.count(r'\d')
    data['uppercase_count'] = data['item_name'].fillna('').apply(
        lambda x: sum(1 for c in str(x) if c.isupper())
    )
    
    # Fill missing values
    if is_train:
        value_median = data['value'].median()
        encodings = {'value_median': value_median}
    else:
        value_median = encodings['value_median']
    
    data['value_filled'] = data['value'].fillna(value_median)
    data['value_missing'] = data['value'].isna().astype(int)
    
    # Value-based features
    data['value_per_pack'] = data['value_filled'] / data['pack_qty']
    data['value_squared'] = data['value_filled'] ** 2
    data['value_log'] = np.log1p(data['value_filled'])
    data['value_sqrt'] = np.sqrt(data['value_filled'])
    data['value_cubed'] = data['value_filled'] ** 3
    data['pack_value_interaction'] = data['pack_qty'] * data['value_filled']
    
    # Premium keywords (expanded) - optimized processing
    catalog_lower = data['catalog_content'].fillna('').astype(str).str.lower()
    
    premium_keywords = {
        'organic': r'organic',
        'premium': r'premium',
        'natural': r'natural',
        'gourmet': r'gourmet',
        'gluten_free': r'gluten.free',
        'non_gmo': r'non.gmo',
        'kosher': r'kosher',
        'vegan': r'vegan',
        'sugar_free': r'sugar.free',
        'family': r'family',
        'luxury': r'luxury|deluxe',
        'artisan': r'artisan',
        'imported': r'imported',
        'certified': r'certified',
        'fresh': r'fresh',
        'pure': r'pure'
    }
    
    for kw_name, pattern in premium_keywords.items():
        data[f'has_{kw_name}'] = catalog_lower.str.contains(pattern, regex=True).astype(int)
    
    # Category detection (optimized)
    catalog_lower = data['catalog_content'].fillna('').astype(str).str.lower()
    
    categories = {
        'food': r'food|snack|meal',
        'beverage': r'beverage|drink|juice',
        'health': r'health|vitamin|supplement',
        'beauty': r'beauty|cosmetic|skin',
        'baby': r'baby|infant',
        'pet': r'pet|dog|cat',
        'household': r'cleaning|detergent|soap'
    }
    
    for cat_name, pattern in categories.items():
        data[f'cat_{cat_name}'] = catalog_lower.str.contains(pattern, regex=True).astype(int)
    
    # Target encoding with K-Fold (prevents leakage)
    if is_train:
        # Use K-Fold for target encoding
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for col in ['unit', 'brand', 'pack_qty']:
            data[f'{col}_mean_price'] = 0.0
            data[f'{col}_median_price'] = 0.0
            data[f'{col}_std_price'] = 0.0
            data[f'{col}_count'] = 0
            
            for train_idx, val_idx in kf.split(data):
                train_fold = data.iloc[train_idx]
                
                # Calculate statistics with smoothing
                global_mean = train_fold['price'].mean()
                smoothing = 10
                
                stats = train_fold.groupby(col)['price'].agg(['mean', 'median', 'std', 'count'])
                
                # Apply smoothed target encoding
                for cat_val in data[col].unique():
                    if cat_val in stats.index:
                        count = stats.loc[cat_val, 'count']
                        mean_val = stats.loc[cat_val, 'mean']
                        smoothed_mean = (mean_val * count + global_mean * smoothing) / (count + smoothing)
                        
                        mask = (data.index.isin(val_idx)) & (data[col] == cat_val)
                        data.loc[mask, f'{col}_mean_price'] = smoothed_mean
                        data.loc[mask, f'{col}_median_price'] = stats.loc[cat_val, 'median']
                        data.loc[mask, f'{col}_std_price'] = stats.loc[cat_val, 'std'] if pd.notna(stats.loc[cat_val, 'std']) else 0
                        data.loc[mask, f'{col}_count'] = count
        
        # Store global statistics for test set
        encodings['global_stats'] = {}
        for col in ['unit', 'brand', 'pack_qty']:
            stats = data.groupby(col)['price'].agg(['mean', 'median', 'std', 'count']).to_dict('index')
            encodings['global_stats'][col] = stats
        encodings['global_mean'] = data['price'].mean()
        encodings['global_median'] = data['price'].median()
    
    else:
        # Apply test set encoding
        global_mean = encodings['global_mean']
        smoothing = 10
        
        for col in ['unit', 'brand', 'pack_qty']:
            stats = encodings['global_stats'][col]
            
            data[f'{col}_mean_price'] = data[col].map(
                lambda x: (stats[x]['mean'] * stats[x]['count'] + global_mean * smoothing) / 
                         (stats[x]['count'] + smoothing) if x in stats else global_mean
            )
            data[f'{col}_median_price'] = data[col].map(
                lambda x: stats[x]['median'] if x in stats else encodings['global_median']
            )
            data[f'{col}_std_price'] = data[col].map(
                lambda x: stats[x]['std'] if x in stats and pd.notna(stats[x]['std']) else 0
            )
            data[f'{col}_count'] = data[col].map(
                lambda x: stats[x]['count'] if x in stats else 0
            )
    
    # Interaction features
    data['value_x_unit_mean'] = data['value_filled'] * data['unit_mean_price']
    data['brand_x_unit_mean'] = data['brand_mean_price'] * data['unit_mean_price']
    data['pack_x_value'] = data['pack_qty'] * data['value_filled']
    data['value_x_brand'] = data['value_filled'] * data['brand_mean_price']
    data['unit_brand_interaction'] = data['unit_mean_price'] * data['brand_mean_price']
    
    # Price per unit estimations
    data['estimated_unit_price'] = data['unit_mean_price'] / (data['value_filled'] + 1)
    data['estimated_brand_premium'] = data['brand_mean_price'] / (encodings['global_mean'] + 1)
    
    # Ratio features
    data['brand_to_unit_ratio'] = data['brand_mean_price'] / (data['unit_mean_price'] + 1)
    data['value_to_pack_ratio'] = data['value_filled'] / (data['pack_qty'] + 1)
    
    return data, encodings

train_df, feature_encodings = extract_features_advanced(train_df, is_train=True)
test_df, _ = extract_features_advanced(test_df, is_train=False, encodings=feature_encodings)

print("Creating enhanced text embeddings...")
# Increase TF-IDF features
vectorizer = TfidfVectorizer(
    max_features=150,  # Increased from 100
    ngram_range=(1, 3),  # Include trigrams
    min_df=3,
    max_df=0.85,
    strip_accents='unicode',
    stop_words='english',
    sublinear_tf=True  # Use sublinear term frequency scaling
)

train_text_features = vectorizer.fit_transform(train_df['item_name'].fillna(''))
test_text_features = vectorizer.transform(test_df['item_name'].fillna(''))

text_df_train = pd.DataFrame(
    train_text_features.toarray(),
    columns=[f'text_{i}' for i in range(train_text_features.shape[1])]
)
text_df_test = pd.DataFrame(
    test_text_features.toarray(),
    columns=[f'text_{i}' for i in range(test_text_features.shape[1])]
)

train_df = pd.concat([train_df.reset_index(drop=True), text_df_train], axis=1)
test_df = pd.concat([test_df.reset_index(drop=True), text_df_test], axis=1)

print("Preparing feature matrix...")
numeric_cols = [
    'value_filled', 'value_missing', 'pack_qty', 'catalog_length', 'name_length', 
    'word_count', 'unique_words', 'avg_word_length', 'unique_ratio', 
    'bullet_points', 'special_chars', 'digit_count', 'uppercase_count',
    'value_per_pack', 'value_squared', 'value_log', 'value_sqrt', 'value_cubed',
    'pack_value_interaction',
    'unit_mean_price', 'unit_median_price', 'unit_std_price', 'unit_count',
    'brand_mean_price', 'brand_median_price', 'brand_std_price', 'brand_count',
    'pack_qty_mean_price', 'pack_qty_median_price', 'pack_qty_std_price', 'pack_qty_count',
    'value_x_unit_mean', 'brand_x_unit_mean', 'pack_x_value', 'value_x_brand',
    'unit_brand_interaction', 'estimated_unit_price', 'estimated_brand_premium',
    'brand_to_unit_ratio', 'value_to_pack_ratio'
]

keyword_cols = [col for col in train_df.columns if col.startswith('has_')]
category_cols = [col for col in train_df.columns if col.startswith('cat_')]
text_cols = [col for col in train_df.columns if col.startswith('text_')]

all_features = numeric_cols + keyword_cols + category_cols + text_cols

X_train = train_df[all_features].fillna(0).replace([np.inf, -np.inf], 0)
y_train = train_df['price']
X_test = test_df[all_features].fillna(0).replace([np.inf, -np.inf], 0)

print(f"Total features: {X_train.shape[1]}")
print(f"  - Numeric: {len(numeric_cols)}")
print(f"  - Keywords: {len(keyword_cols)}")
print(f"  - Categories: {len(category_cols)}")
print(f"  - Text embeddings: {len(text_cols)}\n")

# Apply feature scaling for better convergence
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train = pd.DataFrame(X_train_scaled, columns=all_features)
X_test = pd.DataFrame(X_test_scaled, columns=all_features)

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.12, random_state=42)

def calculate_smape(actual, predicted):
    """Symmetric Mean Absolute Percentage Error"""
    denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
    diff = np.abs(actual - predicted) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)

print("Training optimized models for SMAPE < 5%...")
print("="*70)

# Use log transformation for better SMAPE optimization
y_train_log = np.log1p(y_train)
y_tr_log = np.log1p(y_tr)
y_val_log = np.log1p(y_val)

# Optimized XGBoost parameters - tuned for SMAPE
xgb_params = {
    'max_depth': 8,
    'learning_rate': 0.01,
    'n_estimators': 10000,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'colsample_bylevel': 0.85,
    'min_child_weight': 2,
    'gamma': 0.1,
    'reg_alpha': 0.5,
    'reg_lambda': 1.0,
    'random_state': 42,
    'verbosity': 0,
    'early_stopping_rounds': 400,
    'objective': 'reg:squarederror'
}

if USE_GPU:
    xgb_params.update({'tree_method': 'gpu_hist', 'gpu_id': 0, 'predictor': 'gpu_predictor'})
else:
    xgb_params['n_jobs'] = -1

print("  Training XGBoost...", end=" ", flush=True)
model_xgb = xgb.XGBRegressor(**xgb_params)
model_xgb.fit(X_tr, y_tr_log, eval_set=[(X_val, y_val_log)], verbose=False)
pred_xgb_val_log = model_xgb.predict(X_val)
pred_xgb_val = np.expm1(pred_xgb_val_log)
smape_xgb = calculate_smape(y_val, pred_xgb_val)
print(f"SMAPE: {smape_xgb:.3f}%")

# Optimized LightGBM parameters - tuned for SMAPE
lgb_params = {
    'max_depth': 8,
    'learning_rate': 0.01,
    'n_estimators': 10000,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'min_child_samples': 15,
    'reg_alpha': 0.5,
    'reg_lambda': 1.0,
    'num_leaves': 100,
    'random_state': 42,
    'verbosity': -1,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'min_gain_to_split': 0.01
}

if USE_GPU:
    lgb_params.update({'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0})
else:
    lgb_params['n_jobs'] = -1

print("  Training LightGBM...", end=" ", flush=True)
model_lgb = lgb.LGBMRegressor(**lgb_params)
model_lgb.fit(X_tr, y_tr_log, eval_set=[(X_val, y_val_log)],
              callbacks=[lgb.early_stopping(400), lgb.log_evaluation(0)])
pred_lgb_val_log = model_lgb.predict(X_val)
pred_lgb_val = np.expm1(pred_lgb_val_log)
smape_lgb = calculate_smape(y_val, pred_lgb_val)
print(f"SMAPE: {smape_lgb:.3f}%")

# Optimized CatBoost parameters - tuned for SMAPE
cat_params = {
    'learning_rate': 0.01,
    'depth': 8,
    'l2_leaf_reg': 3,
    'iterations': 10000,
    'random_state': 42,
    'verbose': False,
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.85,
    'border_count': 128,
    'loss_function': 'RMSE'
}

if USE_GPU:
    cat_params.update({'task_type': 'GPU', 'devices': '0'})
else:
    cat_params['task_type'] = 'CPU'
    cat_params['rsm'] = 0.8  # Random subspace method (CPU only)

print("  Training CatBoost...", end=" ", flush=True)
model_cat = CatBoostRegressor(**cat_params)
model_cat.fit(X_tr, y_tr_log, eval_set=(X_val, y_val_log), early_stopping_rounds=400, verbose=False)
pred_cat_val_log = model_cat.predict(X_val)
pred_cat_val = np.expm1(pred_cat_val_log)
smape_cat = calculate_smape(y_val, pred_cat_val)
print(f"SMAPE: {smape_cat:.3f}%")

print("="*70)
print("\nOptimizing ensemble weights with fine-grained search...")
best_score = float('inf')
best_w = (0.33, 0.33, 0.34)

# Fine-grained weight search
for w1 in np.arange(0.25, 0.55, 0.01):
    for w2 in np.arange(0.25, 0.55, 0.01):
        w3 = 1.0 - w1 - w2
        if w3 < 0.15 or w3 > 0.50:
            continue
        ensemble_val = w1 * pred_xgb_val + w2 * pred_lgb_val + w3 * pred_cat_val
        score = calculate_smape(y_val, ensemble_val)
        if score < best_score:
            best_score = score
            best_w = (w1, w2, w3)

print(f"Optimal weights: XGB={best_w[0]:.3f}, LGB={best_w[1]:.3f}, CAT={best_w[2]:.3f}")
print(f"Validation SMAPE: {best_score:.3f}%\n")

if best_score > 5.0:
    print("⚠️  Warning: Validation SMAPE > 5%. Trying stacking approach...")
    
    # Stack predictions as features
    from sklearn.linear_model import Ridge
    stack_features = np.column_stack([pred_xgb_val, pred_lgb_val, pred_cat_val])
    stacker = Ridge(alpha=1.0, random_state=42)
    stacker.fit(stack_features, y_val)
    stacked_pred = stacker.predict(stack_features)
    stacked_smape = calculate_smape(y_val, stacked_pred)
    
    if stacked_smape < best_score:
        print(f"✓ Stacking improved SMAPE to {stacked_smape:.3f}%")
        best_score = stacked_smape
        USE_STACKING = True
    else:
        USE_STACKING = False
else:
    USE_STACKING = False

print("Retraining on full dataset with log transformation...")

xgb_final_params = xgb_params.copy()
xgb_final_params['n_estimators'] = model_xgb.best_iteration
xgb_final_params.pop('early_stopping_rounds', None)

model_xgb_final = xgb.XGBRegressor(**xgb_final_params)
model_xgb_final.fit(X_train, y_train_log, verbose=False)

lgb_final_params = lgb_params.copy()
lgb_final_params['n_estimators'] = model_lgb.best_iteration_

model_lgb_final = lgb.LGBMRegressor(**lgb_final_params)
model_lgb_final.fit(X_train, y_train_log, callbacks=[lgb.log_evaluation(0)])

cat_final_params = cat_params.copy()
cat_final_params['iterations'] = model_cat.best_iteration_

model_cat_final = CatBoostRegressor(**cat_final_params)
model_cat_final.fit(X_train, y_train_log, verbose=False)

print("Generating final predictions...")
pred_xgb_test_log = model_xgb_final.predict(X_test)
pred_lgb_test_log = model_lgb_final.predict(X_test)
pred_cat_test_log = model_cat_final.predict(X_test)

pred_xgb_test = np.expm1(pred_xgb_test_log)
pred_lgb_test = np.expm1(pred_lgb_test_log)
pred_cat_test = np.expm1(pred_cat_test_log)

if USE_STACKING:
    stack_test_features = np.column_stack([pred_xgb_test, pred_lgb_test, pred_cat_test])
    final_predictions = stacker.predict(stack_test_features)
else:
    final_predictions = best_w[0] * pred_xgb_test + best_w[1] * pred_lgb_test + best_w[2] * pred_cat_test

# Ensure positive predictions
final_predictions = np.maximum(final_predictions, 0.01)

output_df = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'price': final_predictions
})

output_df.to_csv('dataset/test_out.csv', index=False)

print("\n" + "="*70)
print("OPTIMIZED TRAINING COMPLETE")
print("="*70)
print(f"✓ Predictions saved: dataset/test_out.csv")
print(f"✓ Total predictions: {len(output_df):,}")
print(f"✓ Price range: ${final_predictions.min():.2f} - ${final_predictions.max():.2f}")
print(f"✓ Mean price: ${final_predictions.mean():.2f}")
print(f"✓ Median price: ${np.median(final_predictions):.2f}")
print(f"✓ Expected SMAPE: ~{best_score:.3f}%")
print("="*70 + "\n")
