import os
import sys
import pandas as pd
import numpy as np
import warnings
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("Smart Product Pricing Model - Training Pipeline")
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

print("Extracting features from product catalog...")

def extract_features(df, is_train=True, encodings=None):
    data = df.copy()
    
    data['value'] = data['catalog_content'].apply(
        lambda x: float(re.search(r'Value:\s*([\d.]+)', str(x)).group(1)) 
        if pd.notna(x) and re.search(r'Value:\s*([\d.]+)', str(x)) else np.nan
    )
    
    data['unit'] = data['catalog_content'].apply(
        lambda x: re.search(r'Unit:\s*([^\n]+)', str(x)).group(1).strip().lower() 
        if pd.notna(x) and re.search(r'Unit:\s*([^\n]+)', str(x)) else 'unknown'
    )
    
    data['item_name'] = data['catalog_content'].apply(
        lambda x: re.search(r'Item Name:\s*([^\n]+)', str(x)).group(1).strip() 
        if pd.notna(x) and re.search(r'Item Name:\s*([^\n]+)', str(x)) else ''
    )
    
    def get_pack_quantity(text):
        patterns = [r'Pack of (\d+)', r'\(Pack of (\d+)\)', r'(\d+)[- ]Pack', r'(\d+) ct']
        for pattern in patterns:
            match = re.search(pattern, str(text), re.I)
            if match:
                return int(match.group(1))
        return 1
    
    data['pack_qty'] = data['catalog_content'].apply(get_pack_quantity)
    
    data['brand'] = data['item_name'].apply(
        lambda x: str(x).split()[0].strip().lower() if len(str(x).split()) > 0 else 'unknown'
    )
    
    data['catalog_length'] = data['catalog_content'].fillna('').str.len()
    data['name_length'] = data['item_name'].fillna('').str.len()
    data['word_count'] = data['item_name'].fillna('').str.split().str.len()
    data['unique_words'] = data['item_name'].fillna('').apply(
        lambda x: len(set(str(x).lower().split()))
    )
    data['avg_word_length'] = data['name_length'] / (data['word_count'] + 1)
    
    data['bullet_points'] = data['catalog_content'].fillna('').str.count('Bullet Point')
    data['special_chars'] = data['item_name'].fillna('').apply(
        lambda x: sum(not c.isalnum() and not c.isspace() for c in str(x))
    )
    
    median_val = data['value'].median() if is_train else encodings['value_median']
    data['value'] = data['value'].fillna(median_val)
    data['value_per_pack'] = data['value'] / data['pack_qty']
    data['value_squared'] = data['value'] ** 2
    data['value_log'] = np.log1p(data['value'])
    
    keywords = ['organic', 'premium', 'natural', 'gourmet', 'gluten.free', 
                'non.gmo', 'kosher', 'vegan', 'sugar.free', 'family']
    for keyword in keywords:
        pattern = keyword.replace('.', '[- ]')
        data[f'has_{keyword}'] = data['catalog_content'].fillna('').str.lower().str.contains(pattern).astype(int)
    
    if is_train:
        unit_stats = data.groupby('unit')['price'].agg(['mean', 'median', 'std', 'count']).to_dict('index')
        brand_stats = data.groupby('brand')['price'].agg(['mean', 'median', 'std', 'count']).to_dict('index')
        pack_stats = data.groupby('pack_qty')['price'].agg(['mean', 'median', 'std', 'count']).to_dict('index')
        
        encodings = {
            'unit_stats': unit_stats,
            'brand_stats': brand_stats,
            'pack_stats': pack_stats,
            'value_median': median_val,
            'global_mean': data['price'].mean(),
            'global_median': data['price'].median()
        }
    
    global_mean = encodings['global_mean']
    smoothing = 10
    
    for col_name, stats_dict in [('unit', 'unit_stats'), ('brand', 'brand_stats'), ('pack_qty', 'pack_stats')]:
        stats = encodings[stats_dict]
        
        data[f'{col_name}_mean_price'] = data[col_name].map(
            lambda x: (stats[x]['mean'] * stats[x]['count'] + global_mean * smoothing) / 
                     (stats[x]['count'] + smoothing) if x in stats else global_mean
        )
        data[f'{col_name}_median_price'] = data[col_name].map(
            lambda x: stats[x]['median'] if x in stats else encodings['global_median']
        )
        data[f'{col_name}_std_price'] = data[col_name].map(
            lambda x: stats[x]['std'] if x in stats and pd.notna(stats[x]['std']) else 0
        )
    
    data['value_x_unit_mean'] = data['value'] * data['unit_mean_price']
    data['brand_x_unit_mean'] = data['brand_mean_price'] * data['unit_mean_price']
    data['pack_x_value'] = data['pack_qty'] * data['value']
    
    return data, encodings

train_df, feature_encodings = extract_features(train_df, is_train=True)
test_df, _ = extract_features(test_df, is_train=False, encodings=feature_encodings)

print("Creating text embeddings...")
vectorizer = TfidfVectorizer(
    max_features=100,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.9,
    strip_accents='unicode',
    stop_words='english'
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
    'value', 'pack_qty', 'catalog_length', 'name_length', 'word_count', 'unique_words',
    'avg_word_length', 'bullet_points', 'special_chars', 'value_per_pack', 'value_squared',
    'value_log', 'unit_mean_price', 'unit_median_price', 'unit_std_price',
    'brand_mean_price', 'brand_median_price', 'brand_std_price',
    'pack_qty_mean_price', 'pack_qty_median_price', 'pack_qty_std_price',
    'value_x_unit_mean', 'brand_x_unit_mean', 'pack_x_value'
]

keyword_cols = [col for col in train_df.columns if col.startswith('has_')]
text_cols = [col for col in train_df.columns if col.startswith('text_')]

all_features = numeric_cols + keyword_cols + text_cols

X_train = train_df[all_features].fillna(0).replace([np.inf, -np.inf], 0)
y_train = train_df['price']
X_test = test_df[all_features].fillna(0).replace([np.inf, -np.inf], 0)

print(f"Total features: {X_train.shape[1]}")
print(f"  - Numeric: {len(numeric_cols)}")
print(f"  - Keywords: {len(keyword_cols)}")
print(f"  - Text embeddings: {len(text_cols)}\n")

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

def calculate_smape(actual, predicted):
    denominator = (np.abs(actual) + np.abs(predicted)) / 2.0
    diff = np.abs(actual - predicted) / denominator
    diff[denominator == 0] = 0.0
    return 100 * np.mean(diff)

print("Training models...")

xgb_params = {
    'max_depth': 13,
    'learning_rate': 0.015,
    'n_estimators': 5000,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'min_child_weight': 1,
    'gamma': 0.3,
    'reg_alpha': 1.0,
    'reg_lambda': 2.0,
    'random_state': 42,
    'verbosity': 0,
    'early_stopping_rounds': 200
}

if USE_GPU:
    xgb_params.update({'tree_method': 'gpu_hist', 'gpu_id': 0, 'predictor': 'gpu_predictor'})
else:
    xgb_params['n_jobs'] = -1

print("  XGBoost...", end=" ", flush=True)
model_xgb = xgb.XGBRegressor(**xgb_params)
model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
pred_xgb_val = model_xgb.predict(X_val)
smape_xgb = calculate_smape(y_val, pred_xgb_val)
print(f"OK SMAPE: {smape_xgb:.2f}%")

lgb_params = {
    'max_depth': 13,
    'learning_rate': 0.015,
    'n_estimators': 5000,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'min_child_samples': 15,
    'reg_alpha': 1.0,
    'reg_lambda': 2.0,
    'num_leaves': 200,
    'random_state': 42,
    'verbosity': -1
}

if USE_GPU:
    lgb_params.update({'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0})
else:
    lgb_params['n_jobs'] = -1

print("  LightGBM...", end=" ", flush=True)
model_lgb = lgb.LGBMRegressor(**lgb_params)
model_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
pred_lgb_val = model_lgb.predict(X_val)
smape_lgb = calculate_smape(y_val, pred_lgb_val)
print(f"OK SMAPE: {smape_lgb:.2f}%")

cat_params = {
    'learning_rate': 0.015,
    'depth': 11,
    'l2_leaf_reg': 5,
    'iterations': 5000,
    'random_state': 42,
    'verbose': False,
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.75
}

if USE_GPU:
    cat_params.update({'task_type': 'GPU', 'devices': '0'})
else:
    cat_params['task_type'] = 'CPU'

print("  CatBoost...", end=" ", flush=True)
model_cat = CatBoostRegressor(**cat_params)
model_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=200, verbose=False)
pred_cat_val = model_cat.predict(X_val)
smape_cat = calculate_smape(y_val, pred_cat_val)
print(f"OK SMAPE: {smape_cat:.2f}%\n")

print("Finding optimal ensemble weights...")
best_score = float('inf')
best_w = (0.33, 0.33, 0.34)

for w1 in np.arange(0.2, 0.6, 0.02):
    for w2 in np.arange(0.1, 0.5, 0.02):
        w3 = 1.0 - w1 - w2
        if w3 < 0.1 or w3 > 0.6:
            continue
        ensemble_val = w1 * pred_xgb_val + w2 * pred_lgb_val + w3 * pred_cat_val
        score = calculate_smape(y_val, ensemble_val)
        if score < best_score:
            best_score = score
            best_w = (w1, w2, w3)

print(f"Ensemble weights: XGBoost={best_w[0]:.2f}, LightGBM={best_w[1]:.2f}, CatBoost={best_w[2]:.2f}")
print(f"Validation SMAPE: {best_score:.2f}%\n")

print("Retraining on full dataset...")

xgb_final_params = xgb_params.copy()
xgb_final_params['n_estimators'] = model_xgb.best_iteration
xgb_final_params.pop('early_stopping_rounds', None)

model_xgb_final = xgb.XGBRegressor(**xgb_final_params)
model_xgb_final.fit(X_train, y_train, verbose=False)

lgb_final_params = lgb_params.copy()
lgb_final_params['n_estimators'] = model_lgb.best_iteration_

model_lgb_final = lgb.LGBMRegressor(**lgb_final_params)
model_lgb_final.fit(X_train, y_train, callbacks=[lgb.log_evaluation(0)])

cat_final_params = cat_params.copy()
cat_final_params['iterations'] = model_cat.best_iteration_

model_cat_final = CatBoostRegressor(**cat_final_params)
model_cat_final.fit(X_train, y_train, verbose=False)

print("Generating predictions...")
pred_xgb_test = model_xgb_final.predict(X_test)
pred_lgb_test = model_lgb_final.predict(X_test)
pred_cat_test = model_cat_final.predict(X_test)

final_predictions = best_w[0] * pred_xgb_test + best_w[1] * pred_lgb_test + best_w[2] * pred_cat_test
final_predictions = np.maximum(final_predictions, 0.01)

output_df = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'price': final_predictions
})

output_df.to_csv('dataset/test_out.csv', index=False)

print("\n" + "="*70)
print("Training Complete")
print("="*70)
print(f"Predictions saved: dataset/test_out.csv")
print(f"Total predictions: {len(output_df):,}")
print(f"Price range: ${final_predictions.min():.2f} - ${final_predictions.max():.2f}")
print(f"Mean price: ${final_predictions.mean():.2f}")
print("="*70 + "\n")
