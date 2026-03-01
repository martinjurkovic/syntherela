import os
from datetime import datetime
from shutil import rmtree

from data.data_generators import generate_real_data, generate_synthetic_data
from syntherela.metrics.multi_table.statistical import (
    CardinalityShapeSimilarity,
)
from syntherela.metrics.single_column.statistical import ChiSquareTest
from syntherela.metrics.single_table.distance import MaximumMeanDiscrepancy
from syntherela.report import Report


def test_report(capsys):
    real_data, metadata = generate_real_data()
    synthetic_data = generate_synthetic_data()

    report = Report(
        real_data,
        synthetic_data,
        metadata,
        'TEST',
        method_name='TEST',
        dataset_name='TEST',
        run_id='TEST',
        validate_metadata=True,
        single_column_metrics=[ChiSquareTest()],
        single_table_metrics=[MaximumMeanDiscrepancy()],
        multi_table_metrics=[CardinalityShapeSimilarity()],
    )

    # Test that the report is generated correctly
    results = report.generate()
    assert 'ChiSquareTest' in report.results['single_column_metrics']
    assert 'MaximumMeanDiscrepancy' in report.results['single_table_metrics']
    assert 'CardinalityShapeSimilarity' in report.results['multi_table_metrics']
    assert type(report.report_datetime) is datetime
    assert report.report_name == 'TEST'

    # Test that the results are printed correctly
    report.print_results()
    captured = capsys.readouterr()
    assert '"multi_table_metrics"' in captured.out

    # Test that the results are saved correctly (same format as save_results)
    report.save_results(path='tests/tmp')
    saved_filename = (
        f'{report.report_name}_'
        f'{report.report_datetime.strftime("%Y-%m-%d_%H-%M-%S")}.json'
    )
    saved_path = os.path.join('tests', 'tmp', saved_filename)
    assert os.path.isfile(saved_path)

    # Test that the results are loaded correctly
    report.load_from_json(path=saved_path)

    assert report.results.keys() == results.keys()

    rmtree('tests/tmp')

    # Test metric instance retrieval
    chisquare = report.get_metric_instance('ChiSquareTest')
    assert type(chisquare) is ChiSquareTest
    try:
        report.get_metric_instance('NonExistentMetric')
    except ValueError as e:
        assert 'NonExistentMetric' in str(e)

    # Test report with no metrics
    report = Report(
        real_data,
        synthetic_data,
        metadata,
        'TEST',
        method_name='TEST',
        dataset_name='TEST',
        run_id='TEST',
        validate_metadata=True,
        single_column_metrics=[],
        single_table_metrics=[],
        multi_table_metrics=[],
    )
    report.generate()
    captured = capsys.readouterr()
    assert 'No single column metrics to run. Skipping.' in captured.out
    assert 'No single table metrics to run. Skipping.' in captured.out
    assert 'No multi table metrics to run. Skipping.' in captured.out

    # Test report with bad and inapplicable metrics
    class BadMetric:
        name = 'BadMetric'

        def is_applicable(self, *args, **kwargs):
            return True

    class InapplicableMetric:
        name = 'InapplicableMetric'

        def is_applicable(self, *args, **kwargs):
            return False

    report = Report(
        real_data,
        synthetic_data,
        metadata,
        'TEST',
        method_name='TEST',
        dataset_name='TEST',
        run_id='TEST',
        validate_metadata=True,
        single_column_metrics=[InapplicableMetric(), BadMetric()],
        single_table_metrics=[InapplicableMetric(), BadMetric()],
        multi_table_metrics=[BadMetric()],
    )
    report.generate()
    captured = capsys.readouterr()

    assert 'There was a problem with metric BadMetric' in captured.out
    assert (
        'There was a problem with metric InapplicableMetric' not in captured.out
    )
