import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder, StandardScaler

class ColumnDropper(BaseEstimator, TransformerMixin):
    '''
    A transformer class to drop specified columns from a DataFrame.

    Attributes:
        to_drop (list): A list of column names to be dropped.

    Methods:
        fit(X, y=None): Fit the transformer to the data. This method does nothing and is only provided to comply with the Scikit-learn API.
        transform(X): Transform the input DataFrame by dropping specified columns.
    '''

    def __init__(self, to_drop):
        self.to_drop = to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.to_drop = [col for col in self.to_drop if col in X.columns]
        return X.drop(columns=self.to_drop)
    

class OneHotFeatureEncoder(BaseEstimator, TransformerMixin):
    '''
    A transformer class for one-hot encoding specified categorical variables.

    Attributes:
        to_encode (list): A list of column names to be one-hot encoded.

    Methods:
        fit(X, y=None): Fit the transformer to the data.
        transform(X): Transform the input DataFrame by one-hot encoding specified columns.
    '''

    def __init__(self, to_encode):
        self.to_encode = to_encode
        self.encoder = OneHotEncoder(drop='first',
                                     sparse_output=False,
                                     dtype=np.int8,
                                     handle_unknown='ignore',
                                     feature_name_combiner='concat')

    def fit(self, X, y=None):
        self.encoder.fit(X[self.to_encode])
        return self

    def transform(self, X):
        X_one_hot = self.encoder.transform(X[self.to_encode])
        one_hot_df = pd.DataFrame(X_one_hot,
                                  columns=self.encoder.get_feature_names_out(self.to_encode))
        X_reset = X.reset_index(drop=True)
        return pd.concat([X_reset.drop(columns=self.to_encode), one_hot_df], axis=1)
    

class StandardFeatureScaler(BaseEstimator, TransformerMixin):
    '''
    A transformer class for standard scaling specified numerical features and retaining feature names.

    Attributes:
        to_scale (list): A list of column names to be scaled.

    Methods:
        fit(X, y=None): Fit the transformer to the data.
        transform(X): Transform the input DataFrame by standard scaling specified columns and retaining feature names.
    '''
    def __init__(self, to_scale):
        self.to_scale = to_scale
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        self.scaler.fit(X[self.to_scale])
        return self

    def transform(self, X):
        X_scaled = self.scaler.transform(X[self.to_scale])
        scaled_df = pd.DataFrame(X_scaled,
                                 columns=self.scaler.get_feature_names_out(self.to_scale))
        X_reset = X.reset_index(drop=True)
        return pd.concat([X_reset.drop(columns=self.to_scale), scaled_df], axis=1)
    
    
class OrdinalFeatureEncoder(BaseEstimator, TransformerMixin):
    '''
    A transformer class for ordinal encoding specified categorical features and retaining feature names.

    Attributes:
        to_encode (dict): A dictionary where keys are column names and values are lists representing the desired category orders.

    Methods:
        fit(X, y=None): Fit the transformer to the data.
        transform(X): Transform the input DataFrame by ordinal encoding specified columns and retaining feature names.
    '''
    def __init__(self, to_encode):
        self.to_encode = to_encode
        self.encoder = OrdinalEncoder(dtype=np.int8, 
                                      categories=[to_encode[col] for col in to_encode])

    def fit(self, X, y=None):
        self.encoder.fit(X[list(self.to_encode.keys())])
        return self

    def transform(self, X):
        X_ordinal = self.encoder.transform(X[list(self.to_encode.keys())])
        ordinal_encoded_df = pd.DataFrame(X_ordinal,
                                          columns=self.encoder.get_feature_names_out(list(self.to_encode.keys())))
        X_reset = X.reset_index(drop=True)
        return pd.concat([X_reset.drop(columns=list(self.to_encode.keys())), ordinal_encoded_df], axis=1)
    

class TargetFeatureEncoder(BaseEstimator, TransformerMixin):
    '''
    A transformer class for target encoding specified categorical variables.

    Attributes:
        to_encode (list): A list of column names to be target encoded.

    Methods:
        fit(X, y=None): Fit the transformer to the data.
        transform(X): Transform the input DataFrame by target encoding specified columns.
    '''

    def __init__(self, to_encode):
        self.to_encode = to_encode
        self.encoder = TargetEncoder()

    def fit(self, X, y):
        self.encoder.fit(X[self.to_encode], y)
        return self

    def transform(self, X):
        X_target = self.encoder.transform(X[self.to_encode])
        target_df = pd.DataFrame(X_target,
                                 columns=self.encoder.get_feature_names_out(self.to_encode))
        X_reset = X.reset_index(drop=True)
        return pd.concat([X_reset.drop(columns=self.to_encode), target_df], axis=1)
    

class FeatureEngineer(BaseEstimator, TransformerMixin):
    '''
    A transformer class for performing feature engineering on churn-related data.

    Methods:
        fit(X, y=None): Fit the transformer to the data. This method does nothing and is only provided to comply with the Scikit-learn API.
        transform(X): Transform the input DataFrame by engineering churn-related features.
    '''

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        discrete_features = ['customer_age', 
                            'dependent_count', 
                            'months_on_book', 
                            'total_relationship_count', 
                            'months_inactive_12_mon', 
                            'contacts_count_12_mon',
                            'total_trans_ct']
        continuous_features = ['credit_limit', 
                               'total_revolving_bal', 
                               'total_amt_chng_q4_q1', 
                               'total_trans_amt',  
                               'total_ct_chng_q4_q1']
        
        X_copy[discrete_features] = X_copy[discrete_features].astype('int32')
        X_copy[continuous_features] = X_copy[continuous_features].astype('float32')
        
        # Construct ratio features.
        X_copy['products_per_dependent'] = (X_copy['total_relationship_count'] / X_copy['dependent_count']).astype('float32')
        X_copy['trans_amt_per_dependent'] = (X_copy['total_trans_amt'] / X_copy['dependent_count']).astype('float32')
        X_copy['trans_ct_per_dependent'] = (X_copy['total_trans_ct'] / X_copy['dependent_count']).astype('float32')
        X_copy['trans_amt_per_products'] = (X_copy['total_trans_amt'] / X_copy['total_relationship_count']).astype('float32')
        X_copy['trans_ct_per_products'] = (X_copy['total_trans_ct'] / X_copy['total_relationship_count']).astype('float32')
        X_copy['avg_trans_amt'] = (X_copy['total_trans_amt'] / X_copy['total_trans_ct']).astype('float32')
        X_copy['credit_util_rate'] = (X_copy['total_revolving_bal'] / X_copy['credit_limit']).astype('float32')
        X_copy['proportion_inactive_months'] = (X_copy['months_inactive_12_mon'] / X_copy['months_on_book']).astype('float32')
        X_copy['products_per_tenure'] = (X_copy['total_relationship_count'] / X_copy['months_on_book']).astype('float32')
        X_copy['products_per_contacts'] = (X_copy['total_relationship_count'] / X_copy['contacts_count_12_mon']).astype('float32')
        X_copy['dependents_per_contacts'] = (X_copy['dependent_count'] / X_copy['contacts_count_12_mon']).astype('float32')
        X_copy['trans_ct_per_contacts'] = (X_copy['total_trans_ct'] / X_copy['contacts_count_12_mon']).astype('float32')
        X_copy['products_per_inactivity'] = (X_copy['total_relationship_count'] / X_copy['months_inactive_12_mon']).astype('float32')
        X_copy['dependents_per_inactivity'] = (X_copy['dependent_count'] / X_copy['months_inactive_12_mon']).astype('float32')
        X_copy['trans_ct_per_inactivity'] = (X_copy['total_trans_ct'] / X_copy['months_inactive_12_mon']).astype('float32')
        X_copy['trans_amt_per_credit_limit'] = (X_copy['total_trans_amt'] / X_copy['credit_limit']).astype('float32')
        X_copy['age_per_tenure'] = (X_copy['customer_age'] / X_copy['months_on_book']).astype('float32')
        X_copy['trans_ct_per_tenure'] = (X_copy['total_trans_ct'] / X_copy['months_on_book']).astype('float32')
        X_copy['trans_amt_per_tenure'] = (X_copy['total_trans_amt'] / X_copy['months_on_book']).astype('float32')
        
        # Replace division by zero values with zero. It will have this 'zero' meaning.
        X_copy = X_copy.replace({np.inf: 0, 
                                 np.nan: 0})
        
        # Construct sum features.
        X_copy['total_spending'] = (X_copy['total_trans_amt'] + X_copy['total_revolving_bal']).astype('int32')
        X_copy['inactivity_contacts'] = (X_copy['contacts_count_12_mon'] + X_copy['months_inactive_12_mon']).astype('int32')
        
        # Map to ordinal education level and income to sum them.
        education_mapping = {
            'Uneducated': 0,
            'High School': 1,
            'College': 2,
            'Graduate': 3,
            'Post-Graduate': 4,
            'Doctorate': 5,
            'Unknown': 0
        }
        
        income_mapping = {
            'Less than $40K': 0,
            '$40K - $60K': 1,
            '$60K - $80K': 2,
            '$80K - $120K': 3,
            '$120K +': 4,
            'Unknown': 0
        }
        
        # Construct interaction between education and income based feature.
        X_copy['education_numeric'] = X_copy['education_level'].map(education_mapping).astype('int32')
        X_copy['income_numeric'] = X_copy['income_category'].map(income_mapping).astype('int32')
        X_copy['education_income_levels'] = (X_copy['education_numeric'] + X_copy['income_numeric']).astype('int32')
        X_copy = X_copy.drop(columns=['education_numeric', 'income_numeric'])
        
        return X_copy
