"""
Feature Engineering Module for Smart Product Pricing Challenge
ML Challenge 2025

This module provides functions to extract and engineer features from:
- Catalog content (text)
- Product names
- Numeric values and units
- Brand information
"""

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')


class ProductFeatureExtractor:
    """Extract and engineer features from product data"""
    
    def __init__(self):
        self.unit_encoder = LabelEncoder()
        self.brand_encoder = LabelEncoder()
        self.tfidf_vectorizer = None
        self.scaler = StandardScaler()
        self.unit_price_map = {}
        self.brand_price_map = {}
        self.fitted = False
        
    def extract_catalog_features(self, df):
        """Extract basic features from catalog content"""
        df = df.copy()
        
        # Extract value (numeric quantity)
        df['value'] = df['catalog_content'].apply(self._extract_value)
        
        # Extract unit
        df['unit'] = df['catalog_content'].apply(self._extract_unit)
        
        # Extract item name
        df['item_name'] = df['catalog_content'].apply(self._extract_item_name)
        
        # Extract brand (first word/phrase from item name)
        df['brand'] = df['item_name'].apply(self._extract_brand)
        
        # Text length features
        df['catalog_length'] = df['catalog_content'].fillna('').str.len()
        df['item_name_length'] = df['item_name'].fillna('').str.len()
        df['item_name_word_count'] = df['item_name'].fillna('').apply(
            lambda x: len(str(x).split())
        )
        
        # Extract pack quantity (Pack of X)
        df['pack_quantity'] = df['catalog_content'].apply(self._extract_pack_quantity)
        
        # Value per unit metrics
        df['value_filled'] = df['value'].fillna(df['value'].median())
        
        return df
    
    def _extract_value(self, catalog_content):
        """Extract numeric value from catalog content"""
        if pd.isna(catalog_content):
            return np.nan
        
        value_match = re.search(r'Value:\s*([\d.]+)', str(catalog_content))
        if value_match:
            try:
                return float(value_match.group(1))
            except:
                return np.nan
        return np.nan
    
    def _extract_unit(self, catalog_content):
        """Extract unit from catalog content"""
        if pd.isna(catalog_content):
            return 'Unknown'
        
        unit_match = re.search(r'Unit:\s*([^\n]+)', str(catalog_content))
        if unit_match:
            unit = unit_match.group(1).strip()
            return unit if unit else 'Unknown'
        return 'Unknown'
    
    def _extract_item_name(self, catalog_content):
        """Extract item name from catalog content"""
        if pd.isna(catalog_content):
            return ''
        
        name_match = re.search(r'Item Name:\s*([^\n]+)', str(catalog_content))
        if name_match:
            return name_match.group(1).strip()
        return ''
    
    def _extract_brand(self, item_name):
        """Extract brand from item name (first word)"""
        if pd.isna(item_name) or item_name == '':
            return 'Unknown'
        
        # Get first word
        words = str(item_name).split()
        if words:
            brand = words[0]
            # Clean brand name
            brand = re.sub(r'[^\w\s]', '', brand)
            return brand if brand else 'Unknown'
        return 'Unknown'
    
    def _extract_pack_quantity(self, catalog_content):
        """Extract pack quantity (Pack of X)"""
        if pd.isna(catalog_content):
            return 1
        
        # Look for "Pack of X" pattern
        pack_match = re.search(r'Pack of (\d+)', str(catalog_content), re.IGNORECASE)
        if pack_match:
            try:
                return int(pack_match.group(1))
            except:
                return 1
        
        # Look for "(Pack of X)" pattern
        pack_match2 = re.search(r'\(Pack of (\d+)\)', str(catalog_content), re.IGNORECASE)
        if pack_match2:
            try:
                return int(pack_match2.group(1))
            except:
                return 1
        
        return 1
    
    def create_aggregated_features(self, df, target_col='price'):
        """Create aggregated features based on categorical variables"""
        df = df.copy()
        
        if target_col in df.columns:
            # Unit-based aggregations
            unit_stats = df.groupby('unit')[target_col].agg(['mean', 'median', 'std', 'count'])
            unit_stats.columns = ['unit_price_mean', 'unit_price_median', 'unit_price_std', 'unit_count']
            df = df.merge(unit_stats, left_on='unit', right_index=True, how='left')
            
            # Brand-based aggregations
            brand_stats = df.groupby('brand')[target_col].agg(['mean', 'median', 'std', 'count'])
            brand_stats.columns = ['brand_price_mean', 'brand_price_median', 'brand_price_std', 'brand_count']
            df = df.merge(brand_stats, left_on='brand', right_index=True, how='left')
            
            # Store mappings for test data
            self.unit_price_map = unit_stats.to_dict()
            self.brand_price_map = brand_stats.to_dict()
        else:
            # Use stored mappings from training
            for col_name, mapping in self.unit_price_map.items():
                df[col_name] = df['unit'].map(mapping)
            
            for col_name, mapping in self.brand_price_map.items():
                df[col_name] = df['brand'].map(mapping)
        
        # Fill missing aggregated features with global mean
        agg_cols = ['unit_price_mean', 'unit_price_median', 'unit_price_std', 'unit_count',
                    'brand_price_mean', 'brand_price_median', 'brand_price_std', 'brand_count']
        for col in agg_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        return df
    
    def create_ratio_features(self, df):
        """Create ratio and interaction features"""
        df = df.copy()
        
        # Price per unit value
        df['value_pack_ratio'] = df['value_filled'] / df['pack_quantity']
        
        # Text complexity
        df['avg_word_length'] = df['item_name_length'] / (df['item_name_word_count'] + 1)
        
        # Unit-brand interaction
        df['unit_brand'] = df['unit'].astype(str) + '_' + df['brand'].astype(str)
        
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """Encode categorical features"""
        df = df.copy()
        
        if fit:
            # Fit encoders on training data
            df['unit_encoded'] = self.unit_encoder.fit_transform(df['unit'].fillna('Unknown'))
            df['brand_encoded'] = self.brand_encoder.fit_transform(df['brand'].fillna('Unknown'))
        else:
            # Transform using fitted encoders
            # Handle unseen categories
            df['unit_encoded'] = df['unit'].fillna('Unknown').apply(
                lambda x: self.unit_encoder.transform([x])[0] 
                if x in self.unit_encoder.classes_ else -1
            )
            df['brand_encoded'] = df['brand'].fillna('Unknown').apply(
                lambda x: self.brand_encoder.transform([x])[0] 
                if x in self.brand_encoder.classes_ else -1
            )
        
        return df
    
    def create_text_features(self, df, fit=True, max_features=100):
        """Create TF-IDF features from item names"""
        df = df.copy()
        
        if fit:
            # Fit TF-IDF on training data
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=5
            )
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(
                df['item_name'].fillna('')
            )
        else:
            # Transform using fitted vectorizer
            if self.tfidf_vectorizer is None:
                raise ValueError("TF-IDF vectorizer not fitted yet!")
            tfidf_matrix = self.tfidf_vectorizer.transform(
                df['item_name'].fillna('')
            )
        
        # Convert to DataFrame
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])],
            index=df.index
        )
        
        df = pd.concat([df, tfidf_df], axis=1)
        
        return df
    
    def fit_transform(self, df, target_col='price', use_tfidf=True, max_tfidf_features=100):
        """Fit and transform training data"""
        print("Extracting catalog features...")
        df = self.extract_catalog_features(df)
        
        print("Creating aggregated features...")
        df = self.create_aggregated_features(df, target_col=target_col)
        
        print("Creating ratio features...")
        df = self.create_ratio_features(df)
        
        print("Encoding categorical features...")
        df = self.encode_categorical_features(df, fit=True)
        
        if use_tfidf:
            print(f"Creating TF-IDF features (max_features={max_tfidf_features})...")
            df = self.create_text_features(df, fit=True, max_features=max_tfidf_features)
        
        self.fitted = True
        print("Feature extraction completed!")
        
        return df
    
    def transform(self, df, use_tfidf=True):
        """Transform test data using fitted extractors"""
        if not self.fitted:
            raise ValueError("Extractor not fitted yet! Call fit_transform first.")
        
        print("Extracting catalog features...")
        df = self.extract_catalog_features(df)
        
        print("Creating aggregated features...")
        df = self.create_aggregated_features(df)
        
        print("Creating ratio features...")
        df = self.create_ratio_features(df)
        
        print("Encoding categorical features...")
        df = self.encode_categorical_features(df, fit=False)
        
        if use_tfidf:
            print("Creating TF-IDF features...")
            df = self.create_text_features(df, fit=False)
        
        print("Feature extraction completed!")
        
        return df
    
    def get_feature_names(self, use_tfidf=True):
        """Get list of feature names for modeling"""
        base_features = [
            'value_filled', 'pack_quantity', 'catalog_length', 'item_name_length',
            'item_name_word_count', 'value_pack_ratio', 'avg_word_length',
            'unit_encoded', 'brand_encoded',
            'unit_price_mean', 'unit_price_median', 'unit_price_std', 'unit_count',
            'brand_price_mean', 'brand_price_median', 'brand_price_std', 'brand_count'
        ]
        
        if use_tfidf and self.tfidf_vectorizer is not None:
            tfidf_features = [f'tfidf_{i}' for i in range(len(self.tfidf_vectorizer.get_feature_names_out()))]
            base_features.extend(tfidf_features)
        
        return base_features


def prepare_features_for_modeling(df, feature_extractor, target_col='price', is_train=True):
    """Prepare final feature matrix for modeling"""
    
    if is_train:
        df_features = feature_extractor.fit_transform(df, target_col=target_col)
    else:
        df_features = feature_extractor.transform(df)
    
    # Get feature columns
    feature_cols = feature_extractor.get_feature_names()
    
    # Handle missing features
    for col in feature_cols:
        if col not in df_features.columns:
            df_features[col] = 0
    
    # Select feature columns
    X = df_features[feature_cols].copy()
    
    # Fill any remaining NaN values
    X = X.fillna(X.median())
    
    # Get target if training
    y = None
    if is_train and target_col in df_features.columns:
        y = df_features[target_col].copy()
    
    return X, y, df_features


# Utility functions
def smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error"""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return np.mean(numerator / denominator) * 100


def evaluate_model(y_true, y_pred):
    """Evaluate model performance with multiple metrics"""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    smape_score = smape(y_true, y_pred)
    
    print(f"MAE: ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"SMAPE: {smape_score:.2f}%")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'smape': smape_score
    }


if __name__ == "__main__":
    print("Feature Engineering Module for Smart Product Pricing Challenge")
    print("=" * 70)
    print("This module provides the ProductFeatureExtractor class for:")
    print("- Extracting features from catalog content")
    print("- Creating aggregated features")
    print("- Encoding categorical variables")
    print("- Creating TF-IDF text features")
    print("=" * 70)
