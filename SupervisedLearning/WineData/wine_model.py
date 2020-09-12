import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBRFClassifier
from xgboost.callback import print_evaluation
from xgboost.callback import early_stop
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


def load_data():
    url_red = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    url_white = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'

    red_df = pd.read_csv(url_red, sep=';')
    white_df = pd.read_csv(url_white, sep=';')
    wine_df = (
        pd.concat([white_df, red_df],
                  axis=0,
                  keys=['Red', 'White'],
                  names=['wine_colour'])
            .reset_index()
    )
    wine_df.drop(labels='level_1',
                 axis=1,
                 inplace=True)
    return wine_df


def split_clean_data(wine_df):
    wine_df["alcohol_cat"] = pd.cut(wine_df["alcohol"],
                                    bins=[7.9, 9.25, 10.5, 11.75, np.inf],
                                    labels=[1, 2, 3, 4, ])
    ss_split = StratifiedShuffleSplit(random_state=42,
                                      test_size=0.2,
                                      n_splits=1)

    for train_idx, test_val_idx in ss_split.split(wine_df, wine_df['alcohol_cat']):
        strat_train_set = wine_df.iloc[train_idx]
        strat_test_val_set = wine_df.iloc[test_val_idx]

    ss_split = StratifiedShuffleSplit(random_state=42,
                                      test_size=0.5,
                                      n_splits=1)

    for test_idx, val_idx in ss_split.split(strat_test_val_set, strat_test_val_set['quality']):
        strat_test_set = strat_test_val_set.iloc[test_idx]
        strat_val_set = strat_test_val_set.iloc[val_idx]

    X_train = strat_train_set.drop('quality',
                                   axis=1)
    y_train = strat_train_set['quality'].apply(fix_quality, )

    cat_attr = ['wine_colour']
    num_attr = [name for name in strat_train_set.drop('quality', axis=1).columns if name not in cat_attr]

    data_transform = ColumnTransformer([
        ('num', StandardScaler(), num_attr),
        ('cat', OneHotEncoder(drop='first'), cat_attr)
    ])

    X_train = data_transform.fit_transform(X_train)

    X_valid = data_transform.transform(strat_val_set.drop('quality',
                                                          axis=1))
    y_valid = strat_val_set['quality'].apply(fix_quality, )

    X_test = data_transform.transform(strat_test_set.drop('quality',
                                                          axis=1))
    y_test = strat_test_set['quality'].apply(fix_quality, )

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def fix_quality(x):
    if x <= 5:
        return 0
    if x == 6:
        return 1
    else:
        return 2


def build_model(X_train, y_train, X_valid, y_valid):
    best_params = {
        'base_score': 2,
        'colsample_bylevel': 0.75,
        'colsample_bynode': 0.57,
        'colsample_bytree': 0.95,
        'gamma': 0.25,
        'learning_rate': 1.7,
        'max_depth': 18,
        'min_child_weight': 0.025,
        'n_estimators': 353,
        'n_jobs': -1,
        'num_class': 3,
        'num_parallel_tree': 105,
        'objective': 'multi:softmax',
        'random_state': 42,
        'subsample': 0.8,
        'verbosity': 0,
        'reg_alpha': 0.05,
        'reg_lambda': 1,
        'rate_drop': 0.5
    }
    best_xgb = XGBRFClassifier(**best_params)

    best_xgb.fit(X_train, y_train,
                 eval_set=[(X_train, y_train),
                           (X_valid, y_valid)],
                 eval_metric=['merror'],
                 early_stopping_rounds=50,
                 callbacks=[print_evaluation(period=5),
                            early_stop(stopping_rounds=15)],
                 verbose=False,)
    return best_xgb


def test_model(model, X_train, y_train, X_valid, y_valid, X_test, y_test):
    print(classification_report(y_pred=model.predict(X_test), y_true=y_test))
    print('Training Score', accuracy_score(y_train, model.predict(X_train)))
    print('Validation Score', accuracy_score(y_valid, model.predict(X_valid)))
    print('Test Score', accuracy_score(y_test, model.predict(X_test)))


if __name__ == '__main__':
    wine_df = load_data()
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = split_clean_data(wine_df)
    model = build_model(X_train, y_train, X_valid, y_valid)
    test_model(model, X_train, y_train, X_valid, y_valid, X_test, y_test)
