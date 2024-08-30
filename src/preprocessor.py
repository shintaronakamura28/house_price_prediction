from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn import set_config


def create_preprocessing_pipeline():
    num_cols = make_column_selector(dtype_include='number')

    num_pipe = Pipeline([
        ('imputer', KNNImputer()),
         ('scaler', StandardScaler())
])

    preprocessor = ColumnTransformer([
        ('numeric',  num_pipe, num_cols),
    ])

    return preprocessor
