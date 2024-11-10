"""
This module provides functions to load example datasets for testing and demonstration
purposes. The datasets included are the German Credit Data and the Titanic dataset.
"""

from io import StringIO

import pandas as pd
import pkg_resources
from scipy.io import arff


def get_credit_data():
    """Load the German Credit Data dataset.

    The dataset is provided in ARFF format and includes various attributes related to
    creditworthiness.

    Returns:
        pandas.DataFrame: A DataFrame containing the credit data.
    """
    s_io = StringIO(
        pkg_resources.resource_string("pysubgroup", "data/credit-g.arff").decode(
            "utf-8"
        )
    )
    data = arff.loadarff(s_io)[0]
    return pd.DataFrame(data)


def get_titanic_data():
    """Load the Titanic dataset.

    The dataset includes information about the passengers on the Titanic,
    such as age, sex, class, and survival status.

    Returns:
        pandas.DataFrame: A DataFrame containing the Titanic data.
    """
    s_io = StringIO(
        pkg_resources.resource_string("pysubgroup", "data/titanic.csv").decode("utf-8")
    )
    return pd.read_csv(s_io, sep="\t", header=0)
