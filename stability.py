import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import LabelEncoder
import openml
from openml import datasets
import warnings
warnings.filterwarnings('ignore')

# Use the specific dataset ID for Crystal-System-Properties-for-Li-ion-batteries
dataset_id = 43433

try:
    # Load the dataset
    dataset = datasets.get_dataset(dataset_id)
    print(f"Successfully loaded dataset: {dataset.name}")
    
    # Load the dataset without specifying a target first
    X_full = dataset.get_data(dataset_format="dataframe")[0]
    print(f"Dataset shape: {X_full.shape}")
    
    # Using Crystal_System as target and Has_Bandstructure as a feature
    target_attribute = "Crystal_System"
    print(f"\nUsing '{target_attribute}' as the target variable")
    
    # Check the unique values in the target
    print(f"Target unique values: {X_full[target_attribute].unique()}")
    
    # Separate features and target
    y = X_full[target_attribute]
    X = X_full.drop(columns=[target_attribute])
    
    # Need to encode the target for classification
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"Encoded target classes: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    # Drop unnecessary columns but keep Has_Bandstructure
    print("Original columns:", X.columns.tolist())
    columns_to_drop = ['Materials_Id', 'Formula', 'Spacegroup']
    X = X.drop(columns=[col for col in columns_to_drop if col in X.columns])
    print(f"After dropping columns, {X.shape[1]} features remain")
    
    # Convert any remaining string columns to numeric (except Has_Bandstructure)
    for col in X.columns:
        if col != 'Has_Bandstructure' and X[col].dtype == 'object':
            try:
                X[col] = pd.to_numeric(X[col])
            except:
                print(f"Could not convert column {col} to numeric, encoding it...")
                X[col] = LabelEncoder().fit_transform(X[col])
    
    # Ensure Has_Bandstructure is encoded properly
    if 'Has_Bandstructure' in X.columns and X['Has_Bandstructure'].dtype == 'object':
        hbs_encoder = LabelEncoder()
        X['Has_Bandstructure'] = hbs_encoder.fit_transform(X['Has_Bandstructure'])
        print(f"Encoded Has_Bandstructure values: {dict(zip(hbs_encoder.classes_, range(len(hbs_encoder.classes_))))}")
    
    print("\n" + "="*50)
    print("METHOD 1: GRADIENT BOOSTING CLASSIFIER")
    print("="*50)
    
    # 1a. Gradient Boosting Classifier on full dataset
    print("\nRunning GBC on full dataset...")
    gbc = GradientBoostingClassifier()
    gbc.fit(X, y_encoded)
    importance_gbc = gbc.feature_importances_
    gbc_ranking = pd.Series(importance_gbc, index=X.columns).sort_values(ascending=False)
    
    print("\nGBC Feature Ranking (full dataset):")
    print(gbc_ranking.head(10))
    
    # 1b. Identify top feature and create reduced dataset
    gbc_top_feature = gbc_ranking.index[0]
    print(f"\nGBC top feature: {gbc_top_feature}")
    X_reduced_gbc = X.drop(columns=[gbc_top_feature])
    
    # 1c. Rerun GBC on reduced dataset
    print("\nRunning GBC on reduced dataset (top GBC feature removed)...")
    gbc_reduced = GradientBoostingClassifier()
    gbc_reduced.fit(X_reduced_gbc, y_encoded)
    importance_gbc_reduced = gbc_reduced.feature_importances_
    gbc_ranking_reduced = pd.Series(importance_gbc_reduced, index=X_reduced_gbc.columns).sort_values(ascending=False)
    
    print("\nGBC Feature Ranking (reduced dataset):")
    print(gbc_ranking_reduced.head(10))
    
    print("\n" + "="*50)
    print("METHOD 2: FEATURE AGGLOMERATION")
    print("="*50)
    
    # 2a. Feature Agglomeration on full dataset
    print("\nRunning FA on full dataset...")
    n_clusters = min(3, X.shape[1])
    fa = FeatureAgglomeration(n_clusters=n_clusters)
    fa.fit(X)
    
    # Calculate importance based on feature variance
    fa_importance = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        fa_importance[i] = X.iloc[:, i].var()
    
    fa_ranking = pd.Series(fa_importance, index=X.columns).sort_values(ascending=False)
    
    print("\nFA Feature Ranking (full dataset):")
    print(fa_ranking.head(10))
    
    # 2b. Identify top feature and create reduced dataset
    fa_top_feature = fa_ranking.index[0]
    print(f"\nFA top feature: {fa_top_feature}")
    X_reduced_fa = X.drop(columns=[fa_top_feature])
    
    # 2c. Rerun FA on reduced dataset
    print("\nRunning FA on reduced dataset (top FA feature removed)...")
    n_clusters_reduced = min(3, X_reduced_fa.shape[1])
    fa_reduced = FeatureAgglomeration(n_clusters=n_clusters_reduced)
    fa_reduced.fit(X_reduced_fa)
    
    # Calculate importance for reduced dataset
    fa_importance_reduced = np.zeros(X_reduced_fa.shape[1])
    for i in range(X_reduced_fa.shape[1]):
        fa_importance_reduced[i] = X_reduced_fa.iloc[:, i].var()
    
    fa_ranking_reduced = pd.Series(fa_importance_reduced, index=X_reduced_fa.columns).sort_values(ascending=False)
    
    print("\nFA Feature Ranking (reduced dataset):")
    print(fa_ranking_reduced.head(10))
    
    print("\n" + "="*50)
    print("METHOD 3: HIGHLY VARIABLE GENE SELECTION")
    print("="*50)
    
    # 3a. HVGS on full dataset
    print("\nRunning HVGS on full dataset...")
    feature_variance = X.var()
    hvgs_ranking = pd.Series(feature_variance, index=X.columns).sort_values(ascending=False)
    
    print("\nHVGS Feature Ranking (full dataset):")
    print(hvgs_ranking.head(10))
    
    # 3b. Identify top feature and create reduced dataset
    hvgs_top_feature = hvgs_ranking.index[0]
    print(f"\nHVGS top feature: {hvgs_top_feature}")
    X_reduced_hvgs = X.drop(columns=[hvgs_top_feature])
    
    # 3c. Rerun HVGS on reduced dataset
    print("\nRunning HVGS on reduced dataset (top HVGS feature removed)...")
    feature_variance_reduced = X_reduced_hvgs.var()
    hvgs_ranking_reduced = pd.Series(feature_variance_reduced, index=X_reduced_hvgs.columns).sort_values(ascending=False)
    
    print("\nHVGS Feature Ranking (reduced dataset):")
    print(hvgs_ranking_reduced.head(10))
    
    print("\n" + "="*50)
    print("METHOD 4: SPEARMAN CORRELATION")
    print("="*50)
    
    # 4a. Spearman correlation on full dataset
    print("\nRunning Spearman correlation on full dataset...")
    X_y = pd.concat([X, pd.Series(y_encoded, index=X.index, name=target_attribute)], axis=1)
    spearman_corr = X_y.corr(method='spearman')
    spearman_ranking = spearman_corr.iloc[-1, :-1].abs().sort_values(ascending=False)
    
    print("\nSpearman Feature Ranking (full dataset):")
    print(spearman_ranking.head(10))
    
    # 4b. Identify top feature and create reduced dataset
    spearman_top_feature = spearman_ranking.index[0]
    print(f"\nSpearman top feature: {spearman_top_feature}")
    X_reduced_spearman = X.drop(columns=[spearman_top_feature])
    
    # 4c. Rerun Spearman on reduced dataset
    print("\nRunning Spearman on reduced dataset (top Spearman feature removed)...")
    X_y_reduced = pd.concat([X_reduced_spearman, pd.Series(y_encoded, index=X_reduced_spearman.index, name=target_attribute)], axis=1)
    spearman_corr_reduced = X_y_reduced.corr(method='spearman')
    spearman_ranking_reduced = spearman_corr_reduced.iloc[-1, :-1].abs().sort_values(ascending=False)
    
    print("\nSpearman Feature Ranking (reduced dataset):")
    print(spearman_ranking_reduced.head(10))
    
    # Summary of top features
    print("\n" + "="*50)
    print("SUMMARY OF TOP FEATURES BY METHOD")
    print("="*50)
    
    top_features = {
        'GBC': gbc_top_feature,
        'FA': fa_top_feature,
        'HVGS': hvgs_top_feature,
        'Spearman': spearman_top_feature
    }
    
    print("\nTop feature identified by each method:")
    for method, feature in top_features.items():
        print(f"  {method}: {feature}")
    
    print("\nTop 5 features before and after removing the top feature:")
    
    # GBC
    print("\nGBC top 5 before:")
    print(gbc_ranking.head(5))
    print("\nGBC top 5 after removing", gbc_top_feature + ":")
    print(gbc_ranking_reduced.head(5))
    
    # FA
    print("\nFA top 5 before:")
    print(fa_ranking.head(5))
    print("\nFA top 5 after removing", fa_top_feature + ":")
    print(fa_ranking_reduced.head(5))
    
    # HVGS
    print("\nHVGS top 5 before:")
    print(hvgs_ranking.head(5))
    print("\nHVGS top 5 after removing", hvgs_top_feature + ":")
    print(hvgs_ranking_reduced.head(5))
    
    # Spearman
    print("\nSpearman top 5 before:")
    print(spearman_ranking.head(5))
    print("\nSpearman top 5 after removing", spearman_top_feature + ":")
    print(spearman_ranking_reduced.head(5))

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
