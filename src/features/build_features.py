"""Feature engineering functions for credit risk modeling."""
import numpy as np
import pandas as pd


def add_ratio_features(df):
    """Financial ratio features from amounts and time."""
    df = df.copy()
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['CREDIT_TERM_MONTHS'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['EMPLOYED_AGE_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['CREDIT_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['INCOME_PER_FAMILY'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['INCOME_PER_CHILD'] = df['AMT_INCOME_TOTAL'] / (df['CNT_CHILDREN'] + 1)
    return df


def add_ext_source_features(df):
    """Aggregate external credit scores."""
    df = df.copy()
    ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    df['EXT_SOURCE_MEAN'] = df[ext_cols].mean(axis=1)
    df['EXT_SOURCE_STD'] = df[ext_cols].std(axis=1)
    df['EXT_SOURCE_MIN'] = df[ext_cols].min(axis=1)
    df['EXT_SOURCE_MAX'] = df[ext_cols].max(axis=1)
    df['EXT_SOURCE_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    return df


def add_document_count(df):
    """Count of documents submitted by applicant."""
    df = df.copy()
    doc_cols = [c for c in df.columns if c.startswith('FLAG_DOCUMENT_')]
    df['DOCUMENT_COUNT'] = df[doc_cols].sum(axis=1)
    return df


def handle_special_values(df):
    """Flag and fix known placeholder values."""
    df = df.copy()
    df['DAYS_EMPLOYED_ANOMALY'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
    df.loc[df['DAYS_EMPLOYED'] == 365243, 'DAYS_EMPLOYED'] = np.nan
    return df


def build_features(df):
    """Apply all feature engineering steps in order."""
    df = handle_special_values(df)
    df = add_ratio_features(df)
    df = add_ext_source_features(df)
    df = add_document_count(df)
    return df