import numpy as np
import pandas as pd

from syntherela.metadata import Metadata

PROB_ABC = [0.3, 0.5, 0.2]
PROB_123 = [0.1, 0.6, 0.3]
EPS = 1e-6


def generate_real_data(seed=0):
    """
    Generate real data for testing.
    """
    np.random.seed(seed)

    tables = dict()
    tables["table1"] = pd.DataFrame(
        {
            "pk1": np.arange(10),
            "normal": np.random.rand(10),
            "uniform": np.random.uniform(0, 1, 10),
            "categorical": np.random.choice(["a", "b", "c"], 10, p=PROB_ABC),
        }
    )
    tables["table2"] = pd.DataFrame(
        {
            "pk2": np.arange(100),
            "fk2": np.random.choice(np.arange(10), 100),
            "lognormal": np.random.lognormal(0, 1, 100),
            "categorical": np.random.choice(["1", "2", "3"], 100, p=PROB_123),
        }
    )
    # Add dependence between tables
    tables["table2"]["dependent"] = (
        (tables["table1"]["uniform"][tables["table2"]["fk2"]] ** 2).values
        - 0.1 * tables["table2"]["lognormal"].values
        + np.random.normal(0, 1, 100) * EPS
    )

    metadata = Metadata()
    metadata.add_table("table1")
    metadata.add_column("table1", "pk1", sdtype="id")
    metadata.add_column("table1", "normal", sdtype="numerical")
    metadata.add_column("table1", "uniform", sdtype="numerical")
    metadata.add_column("table1", "categorical", sdtype="categorical")
    metadata.set_primary_key("table1", "pk1")

    metadata.add_table("table2")
    metadata.add_column("table2", "pk2", sdtype="id")
    metadata.add_column("table2", "fk2", sdtype="id")
    metadata.add_column("table2", "lognormal", sdtype="numerical")
    metadata.add_column("table2", "categorical", sdtype="categorical")
    metadata.add_column("table2", "dependent", sdtype="numerical")
    metadata.set_primary_key("table2", "pk2")

    metadata.add_relationship("table1", "table2", "pk1", "fk2")
    metadata.validate()
    metadata.validate_data(tables)
    return tables, metadata


def generate_synthetic_data(good_fit=True, seed=0):
    """
    Generate synthetic data for testing.
    """
    np.random.seed(seed)

    if good_fit:
        p_abc = PROB_ABC
        p_123 = PROB_123
        eps = EPS
    else:
        p_abc = [0.2, 0.5, 0.3]
        p_123 = [0.12, 0.5, 0.38]
        eps = EPS * 10

    tables = dict()
    tables["table1"] = pd.DataFrame(
        {
            "pk1": np.arange(100),
            "normal": np.random.rand(100),
            "uniform": np.random.uniform(0, 1, 100),
            "categorical": np.random.choice(["a", "b", "c"], 100, p=p_abc),
        }
    )
    tables["table2"] = pd.DataFrame(
        {
            "pk2": np.arange(1000),
            "fk2": np.random.choice(np.arange(100), 1000),
            "lognormal": np.random.lognormal(0, 1, 1000),
            "categorical": np.random.choice(["1", "2", "3"], 1000, p=p_123),
        }
    )
    # Add dependence between tables
    tables["table2"]["dependent"] = (
        (tables["table1"]["uniform"][tables["table2"]["fk2"]] ** 2).values
        - 0.1 * tables["table2"]["lognormal"].values
        + np.random.normal(0, 1, 1000) * eps
    )

    if not good_fit:
        tables["table1"]["normal"] = np.random.rand(100) * 1.1

    return tables
