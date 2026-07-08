"""Tests for single-table ML utility metric."""

import pytest
from sklearn.ensemble import RandomForestClassifier
from syntherela.metrics.utility import MachineLearningEfficacyMetric


def test_utility_init():
    m = MachineLearningEfficacyMetric(
        target=('table1', 'categorical', None),
        classifier_cls=RandomForestClassifier,
        classifier_args={},
        random_state=0,
    )
    assert m.name is not None and 'MachineLearningEfficacyMetric' in m.name
    assert m.target == ('table1', 'categorical', None)


def test_utility_prepare_data(sample_data):
    data, metadata = sample_data
    m = MachineLearningEfficacyMetric(
        target=('table1', 'categorical', None),
        classifier_cls=RandomForestClassifier,
        classifier_args={},
        random_state=0,
    )
    X = data['table1'].drop(columns=['categorical'])
    X_prep, ht = m.prepare_data(X)
    assert X_prep is not None
    assert ht is not None
    assert len(X_prep) == len(X)


def test_utility_score_classifier(sample_data):
    data, _ = sample_data
    m = MachineLearningEfficacyMetric(
        target=('table1', 'categorical', None),
        classifier_cls=RandomForestClassifier,
        classifier_args={'n_estimators': 5},
        random_state=0,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = data['table1'].drop(columns=['categorical'])
    y = data['table1']['categorical'].astype('category').cat.codes
    X_prep, _ = m.prepare_data(X)
    model = Pipeline(
        [
            ('imputer', SimpleImputer()),
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=5)),
        ]
    )
    model.fit(X_prep, y)
    sc = m.score(model, X_prep, y)
    assert 0 <= sc <= 1


def test_utility_get_target_table(sample_data):
    data, metadata = sample_data
    m = MachineLearningEfficacyMetric(
        target=('table1', 'normal', None),
        classifier_cls=RandomForestClassifier,
        classifier_args={},
        random_state=0,
    )
    X, y = m.get_target_table(data, m.target, metadata)
    assert 'pk1' not in X.columns
    assert 'normal' not in X.columns
    assert len(y) == len(data['table1'])
    assert y.name == 'normal'


def test_utility_run(sample_data):
    data, metadata = sample_data
    synthetic = {
        'table1': data['table1'].copy(),
        'table2': data['table2'].copy(),
    }
    synthetic['table1']['normal'] = synthetic['table1']['normal'] + 0.1
    test_data = {
        'table1': data['table1'].iloc[:5].copy(),
        'table2': data['table2'].iloc[:15].copy(),
    }
    m = MachineLearningEfficacyMetric(
        target=('table1', 'categorical', None),
        classifier_cls=RandomForestClassifier,
        classifier_args={'n_estimators': 5},
        random_state=0,
    )
    result = m.run(
        data, synthetic, metadata, test_data, m=2, feature_importance=False
    )
    assert 'real_score' in result
    assert 'synthetic_score' in result
    assert 'difference' in result
    assert 'synthetic_score_se' in result


def test_utility_feature_importance(sample_data):
    data, metadata = sample_data
    m = MachineLearningEfficacyMetric(
        target=('table1', 'categorical', None),
        classifier_cls=RandomForestClassifier,
        classifier_args={'n_estimators': 5},
        random_state=0,
    )
    synthetic = {k: v.copy() for k, v in data.items()}
    test_data = {
        'table1': data['table1'].iloc[:5],
        'table2': data['table2'].iloc[:15],
    }
    result = m.run(
        data, synthetic, metadata, test_data, m=2, feature_importance=True
    )
    assert 'feature_importance_real' in result
    assert 'feature_importance_synthetic' in result
    assert 'feature_names' in result


def test_utility_feature_importance_raises_for_unsupported(sample_data):
    from sklearn.impute import SimpleImputer
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    data, _ = sample_data
    m = MachineLearningEfficacyMetric(
        target=('table1', 'categorical', None),
        classifier_cls=KNeighborsClassifier,
        classifier_args={},
        random_state=0,
    )
    X = data['table1'].drop(columns=['categorical'])
    y = data['table1']['categorical'].astype('category').cat.codes
    X_prep, _ = m.prepare_data(X)
    m.X_real = X_prep
    m.y_real = y
    model = Pipeline(
        [
            ('imputer', SimpleImputer()),
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier()),
        ]
    )
    model.fit(X_prep, y)
    with pytest.raises(NotImplementedError):
        m.feature_importance(model)
        m.feature_importance(model)
        m.feature_importance(model)
