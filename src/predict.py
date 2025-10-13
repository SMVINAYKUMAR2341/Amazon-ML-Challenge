"""
Prediction Pipeline for Smart Product Pricing Challenge
ML Challenge 2025

This script:
1. Loads trained models and feature extractor
2. Loads test data
3. Extracts features from test data
4. Generates predictions using ensemble of models
5. Saves predictions in the required format (test_out.csv)
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import feature engineering module
from feature_engineering import ProductFeatureExtractor, prepare_features_for_modeling


def load_models(models_dir='../models'):
    """Load trained models and feature extractor"""
    print("Loading models...")
    
    try:
        xgb_model = joblib.load(os.path.join(models_dir, 'xgboost_model.pkl'))
        lgb_model = joblib.load(os.path.join(models_dir, 'lightgbm_model.pkl'))
        cat_model = joblib.load(os.path.join(models_dir, 'catboost_model.pkl'))
        feature_extractor = joblib.load(os.path.join(models_dir, 'feature_extractor.pkl'))
        
        print("✓ Models loaded successfully!")
        return xgb_model, lgb_model, cat_model, feature_extractor
    
    except FileNotFoundError as e:
        print(f"Error: Model files not found in {models_dir}")
        print("Please train models first using 02_model_training.ipynb")
        raise e


def load_test_data(dataset_folder='../dataset'):
    """Load test dataset"""
    print("\nLoading test data...")
    
    test_path = os.path.join(dataset_folder, 'test.csv')
    test = pd.read_csv(test_path)
    
    print(f"✓ Test data loaded: {test.shape}")
    print(f"  Samples: {len(test):,}")
    print(f"  Columns: {test.columns.tolist()}")
    
    return test


def prepare_test_features(test_df, feature_extractor):
    """Prepare features for test data"""
    print("\nPreparing test features...")
    print("=" * 60)
    
    X_test, _, test_features = prepare_features_for_modeling(
        test_df,
        feature_extractor,
        is_train=False
    )
    
    print("=" * 60)
    print(f"✓ Feature matrix shape: {X_test.shape}")
    
    return X_test, test_features


def generate_predictions(X_test, xgb_model, lgb_model, cat_model, ensemble_weights=None):
    """Generate predictions using ensemble of models"""
    print("\nGenerating predictions...")
    
    # Default weights (equal weighting)
    if ensemble_weights is None:
        ensemble_weights = {
            'xgb': 0.33,
            'lgb': 0.33,
            'cat': 0.34
        }
    
    # Predict with each model
    print("  Predicting with XGBoost...")
    pred_xgb = xgb_model.predict(X_test)
    
    print("  Predicting with LightGBM...")
    pred_lgb = lgb_model.predict(X_test)
    
    print("  Predicting with CatBoost...")
    pred_cat = cat_model.predict(X_test)
    
    # Ensemble predictions
    print("  Creating ensemble predictions...")
    predictions = (
        ensemble_weights['xgb'] * pred_xgb +
        ensemble_weights['lgb'] * pred_lgb +
        ensemble_weights['cat'] * pred_cat
    )
    
    # Ensure all predictions are positive
    predictions = np.maximum(predictions, 0.01)
    
    print(f"✓ Predictions generated: {len(predictions):,}")
    print(f"  Min price: ${predictions.min():.2f}")
    print(f"  Max price: ${predictions.max():.2f}")
    print(f"  Mean price: ${predictions.mean():.2f}")
    print(f"  Median price: ${np.median(predictions):.2f}")
    
    return predictions


def save_predictions(test_df, predictions, output_path):
    """Save predictions in the required format"""
    print(f"\nSaving predictions to {output_path}...")
    
    # Create output DataFrame
    output_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': predictions
    })
    
    # Verify format
    assert len(output_df) == len(test_df), "Number of predictions doesn't match test samples!"
    assert output_df['sample_id'].equals(test_df['sample_id']), "Sample IDs don't match!"
    assert output_df['price'].isna().sum() == 0, "Found NaN values in predictions!"
    
    # Save to CSV
    output_df.to_csv(output_path, index=False)
    
    print(f"✓ Predictions saved successfully!")
    print(f"  Total predictions: {len(output_df):,}")
    print(f"  Output file: {output_path}")
    
    # Display sample predictions
    print("\nSample predictions:")
    print(output_df.head(10))
    
    return output_df


def main():
    """Main prediction pipeline"""
    print("=" * 80)
    print("SMART PRODUCT PRICING CHALLENGE - PREDICTION PIPELINE")
    print("ML Challenge 2025")
    print("=" * 80)
    
    # Configuration
    MODELS_DIR = '../models'
    DATASET_DIR = '../dataset'
    OUTPUT_PATH = os.path.join(DATASET_DIR, 'test_out.csv')
    
    # Ensemble weights (tune these based on validation performance)
    ENSEMBLE_WEIGHTS = {
        'xgb': 0.33,
        'lgb': 0.33,
        'cat': 0.34
    }
    
    try:
        # Step 1: Load models
        xgb_model, lgb_model, cat_model, feature_extractor = load_models(MODELS_DIR)
        
        # Step 2: Load test data
        test_df = load_test_data(DATASET_DIR)
        
        # Step 3: Prepare features
        X_test, test_features = prepare_test_features(test_df, feature_extractor)
        
        # Step 4: Generate predictions
        predictions = generate_predictions(
            X_test, xgb_model, lgb_model, cat_model, 
            ensemble_weights=ENSEMBLE_WEIGHTS
        )
        
        # Step 5: Save predictions
        output_df = save_predictions(test_df, predictions, OUTPUT_PATH)
        
        print("\n" + "=" * 80)
        print("PREDICTION PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nOutput file ready for submission: {OUTPUT_PATH}")
        print("\nNext steps:")
        print("  1. Verify the output format matches sample_test_out.csv")
        print("  2. Upload test_out.csv to the submission portal")
        print("  3. Complete the documentation (Documentation_template.md)")
        print("=" * 80)
        
        return output_df
    
    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR IN PREDICTION PIPELINE")
        print("=" * 80)
        print(f"\nError: {str(e)}")
        print("\nPlease ensure:")
        print("  1. Models are trained (run 02_model_training.ipynb)")
        print("  2. Test data exists in dataset/ folder")
        print("  3. All required packages are installed")
        print("=" * 80)
        raise e


if __name__ == "__main__":
    main()
