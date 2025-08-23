import warnings

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.exceptions import UndefinedMetricWarning
from statsmodels.stats.multitest import multipletests

import pysubgroup as ps


class NegativeClassCountRandomSelector(ps.SelectorBase):
    """
    A selector that covers a random subset of the given indices, such that
    the number of covered instances as well as the number of negatives instances
    are always the same.
    """

    def __init__(
        self,
        size,
        negative_class_count,
        np_rng,
        positive_class_indices,
        negative_class_indices,
    ):
        self.size = size
        self.negative_class_count = negative_class_count
        self.np_rng = np_rng
        self.positive_class_indices = positive_class_indices
        self.negative_class_indices = negative_class_indices

        self.select()

        self._hash = None
        self._query = None
        self._string = None
        self.set_descriptions(size, negative_class_count)

        super().__init__()

    def select(self):
        """Randomize the cover of this selector."""
        # Results are equivalent to randomly permuting the negatives and positives independent from each other
        # and then retrieving the cover of the subgroup from the data again.
        included_positive_class_indices = self.np_rng.choice(
            self.positive_class_indices,
            self.size - self.negative_class_count,
            replace=False,
        )
        included_negative_class_indices = self.np_rng.choice(
            self.negative_class_indices, self.negative_class_count, replace=False
        )
        self._included_indices = [
            *included_positive_class_indices,
            *included_negative_class_indices,
        ]

    def covers(self, data_instance):
        return data_instance.index.isin(self._included_indices)

    def set_descriptions(self, size, negative_class_count, *args, **kwargs):
        query = f"{size} rows with negative class count {negative_class_count} randomly sampled without replacement"
        self._hash, self._query, self._string = (hash(query), query, query)

    @property
    def selectors(self):
        return (self,)


def _compute_random_sample_value(selector, qf, data, target, max_retries: int = 10):
    """
    Randomize a selector until its quality is defined, with a maximum number of tries.
    Throw an exception when the maximum number of tries is surpassed.

    :param selector: The selector to evaluate the quality of. Must implement a function .select() which implements the rerandomization of its cover.
    :param qf: The quality function to compute the quality from.
    :param data: The dataset to take the cover of the selector from.
    :param target: The definition of the target concept to tell the quality function which attributes to evaluate the cover on.
    :max_retries: The limit on the number of rerandomizations of the selector. An exception is thrown when this is surpassed.
    """
    quality = np.nan
    retries = -1

    while np.isnan(quality):
        if retries >= max_retries:
            raise Exception(
                "Random sampling exceeded max retries ({self.max_retries}). "
                + "Make sure that the subgroup is well represented in the "
                + "testing data and increase max_retries."
            )

        selector.select()

        subgroup = ps.create_subgroup_with_representation(data, [selector])
        statistics = qf.calculate_statistics(subgroup, target, data)
        quality = qf.evaluate(subgroup, target, data, statistics)

        retries += 1

    return quality


def _random_sampling(
    qf: any,
    subgroup_labels: ArrayLike,
    target: ps.SoftClassifierTarget,
    data: pd.DataFrame,
    num_random_samples: int,
    np_rng: np.random.Generator,
    max_retries: int = 10,
    pos_label: any = 1,
    neg_label: any = 0,
):
    """
    Compute the values of many randomly sampled data subsets, that all have
    the same number of positive and negative instances as in the given subgroup labels.
    """
    negatives_count = np.sum(subgroup_labels.to_numpy() == neg_label)

    positive_class_indices = data[data[target.label_column] == pos_label].index
    negative_class_indices = data[data[target.label_column] == neg_label].index

    # Disable warnings about undefined metrics from sklearn.
    # Undefined qualities are covered by the behaviour of the sampling process, since it includes retries in that case.
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

    random_selector = NegativeClassCountRandomSelector(
        len(subgroup_labels),
        negatives_count,
        np_rng,
        positive_class_indices,
        negative_class_indices,
    )
    sample_values = [
        _compute_random_sample_value(random_selector, qf, data, target, max_retries)
        for selector in range(num_random_samples)
    ]

    # Reenable warnings in case they are needed elsewhere.
    warnings.filterwarnings("default", category=UndefinedMetricWarning)

    return sample_values


def permutation_test(
    qf: any,
    result: ps.SubgroupDiscoveryResult,
    target: ps.SoftClassifierTarget,
    data: pd.DataFrame,
    num_random_samples: int,
    np_rng: np.random.Generator = np.random.default_rng(),
    max_random_sampling_retries: int = 10,
    alpha: float = 0.05,
    pos_label: any = 1,
    neg_label: any = 0,
    multitest_correction_method: str = "fdr_by",
    tqdm: any = None,
) -> tuple[list[float], list[float], list[list[float]]]:
    """
    Test the subgroups in the result for statistical significance by comparison to qualities of random samples from the data.
    Random samples are drawn such that the number of instances from each class in the sample is the same as in the tested subgroup.

    Only for SoftClassifierTargets.

    :param qf: Quality function to use as the test statistic.
    :param result: Result object holding the subgroups to test.
    :param target: Target concept to use in the quality function.
    :param data: Dataset to compute all qualities from. The qualities of the given subgroups are also recomputed on this data for the test.
    :param num_random_samples: How many random samples to draw. More samples allow to distinguish p-values more fine-grained.
    :param np_rng: Random generator object to use for drawing the samples. Use this to get reproducible results.
    :param max_random_sampling_retries: How often to repeat the drawing process for each sample to get a quality. Repetitions are used when the quality is undefined on a random sample.
    :param pos_label: Which value in the dataset to count as a positive class.
    :param neg_label: Which value in the dataset to count as a negative class.
    :param multitest_correction_method: Which method to correct the p-values against the multiple comparison problem with. Refer to statsmodels.stats.multitest.multipletests for all possible values.
    """
    if tqdm is None:
        tqdm = lambda x: x

    p_values_raw = []
    qualities = []
    samples = []

    for result_item in tqdm(result.results):
        subgroup = result_item[1]
        subgroup = ps.create_subgroup_with_representation(data, subgroup.selectors)
        sg_labels = data[subgroup.representation][target.label_column]

        qf_value = qf.evaluate(subgroup, target, data)
        qualities.append(qf_value)

        if qf_value == -np.inf:
            print(
                f"skipping subgroup, because its quality undefined on the given data: {subgroup}"
            )

            p_values_raw.append(None)
            samples.append(None)

            continue

        sample = _random_sampling(
            qf=qf,
            subgroup_labels=sg_labels,
            target=target,
            data=data,
            num_random_samples=num_random_samples,
            np_rng=np_rng,
            max_retries=max_random_sampling_retries,
            pos_label=pos_label,
            neg_label=neg_label,
        )
        num_at_least_as_extreme_values = sum(
            [sample_value >= qf_value for sample_value in sample]
        )
        p_val = num_at_least_as_extreme_values / num_random_samples

        samples.append(sample)
        p_values_raw.append(p_val)

    # Replacing missing p-values with 1.0 to regard subgroups that could not
    # be tested as hypotheses that could not be rejected.
    p_values_raw = [(p if p is not None else 1.0) for p in p_values_raw]

    # Correcting p-values for multiple comparison problem.
    reject, p_values_corrected, _, _ = multipletests(
        p_values_raw, alpha=alpha, method=multitest_correction_method
    )

    return p_values_raw, reject, p_values_corrected, qualities, samples
