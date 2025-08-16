"""
Created on 16.08.2025

@author: Tom Siegl
"""
from typing import Literal

import numpy as np
import pandas as pd
from sklearn import metrics

import pysubgroup as ps

##########
# target #
##########


class SoftClassifierTarget:
    def __init__(self, label_column="label", prediction_column="prediction"):
        self.label_column = label_column
        self.prediction_column = prediction_column

    def get_target_columns(self, data: pd.DataFrame):
        return data[:, [self.label_column, self.prediction_column]]


########################
# performance measures #
########################


def average_ranking_loss(y_true, y_pred):
    """
    :param y_true: Binary Labels, must be ordered to match y_pred
    :param y_pred: Predicted Scores, must be in ascending order
    """
    negatives_loop_count = 0
    penalty_sum = 0

    last_score = np.inf
    current_positives_count = 0
    current_negatives_count = 0
    for current_gt, current_score in zip(reversed(y_true), reversed(y_pred)):
        if current_score < last_score:
            penalty_sum += (
                2 * negatives_loop_count + current_negatives_count
            ) * current_positives_count
            last_score = current_score
            negatives_loop_count += current_negatives_count
            current_negatives_count = 0
            current_positives_count = 0

        if current_gt:
            current_positives_count += 1
        else:
            current_negatives_count += 1

    penalty_sum += 2 * negatives_loop_count * current_positives_count
    penalty_sum += current_negatives_count * current_positives_count
    negatives_loop_count += current_negatives_count
    positives_count = len(y_true) - negatives_loop_count

    arl = penalty_sum / (2 * positives_count)

    return arl


def pr_auc_score(y_true, y_pred):
    precision, recall, _ = metrics.precision_recall_curve(
        y_true, y_pred, drop_intermediate=True
    )
    return metrics.auc(recall, precision)


##################################
# bounds of performance measures #
##################################


def _contains_tie(y_true: np.array, y_pred: np.array) -> bool:
    """
    Returns True if two different indices with the same y_pred value have different y_true values.
    """
    previous_true = y_true[0]
    previous_pred = y_pred[0]

    for i in range(len(y_true)):
        if y_pred[i] != previous_pred:
            previous_true = y_true[i]
            previous_pred = y_pred[i]
        elif y_true[i] != previous_true:
            return True

    return False


def _contains_error(y_true: np.array, y_pred: np.array) -> bool:
    """
    Returns True if an index with y_true=1 is followed by an index with y_true=0 and higher y_pred.
    """
    true_found = False
    true_pred = None

    for i in range(len(y_true)):
        if (not true_found) and y_true[i]:
            true_found = True
            true_pred = y_pred[i]
        if true_found and (not y_true[i]) and true_pred < y_pred[i]:
            return True

    return False


def _ARL_upper_bound(y_true: np.array, y_pred: np.array) -> float:
    """
    Upper bound of the Average Ranking Loss (ARL) performance measure
    """
    max_pen = 0
    positive_found_score = None

    for label, score in zip(y_true, y_pred):
        if positive_found_score is None and label:
            # first positive instance found -> start counting negatives from here
            positive_found_score = score

        if positive_found_score is not None and not label:
            # negative instance after positive instance found -> add to penalty
            if positive_found_score == score:
                max_pen += 0.5
            else:
                max_pen += 1

    return max_pen


def _ROC_AUC_lower_bound(y_true: np.array, y_pred: np.array) -> float:
    """
    Lower bound of the ROC AUC performance measure
    """
    if len(y_pred) == 0:
        return 0

    if _contains_error(y_true, y_pred):
        return 0

    if _contains_tie(y_true, y_pred):
        return 0.5

    return 1


def _PR_AUC_lower_bound(y_true: np.array, y_pred: np.array) -> float:
    """
    Lower bound of the PR AUC performance measure
    """
    if len(y_pred) == 0:
        return 0

    if (not _contains_error(y_true, y_pred)) and (not _contains_tie(y_true, y_pred)):
        return 1

    worst_subset_y_true = []
    worst_subset_y_pred = []
    positive_found = False
    previous_negative_found = 0
    previous_negative_found_score = None

    for label, score in zip(y_true, y_pred):
        if not positive_found and not label:
            if (
                previous_negative_found_score is None
                or previous_negative_found_score != score
            ):
                previous_negative_found_score = score
                previous_negative_found = 1
            else:
                previous_negative_found += 1

        if not positive_found and label:
            positive_found = True
            worst_subset_y_true.append(label)
            worst_subset_y_pred.append(score)

            # add previous tied negatives
            if score == previous_negative_found_score:
                for i in range(previous_negative_found):
                    worst_subset_y_true.append(0)
                    worst_subset_y_pred.append(previous_negative_found_score)

        if positive_found and not label:
            worst_subset_y_true.append(label)
            worst_subset_y_pred.append(score)

    return prc_auc_score(worst_subset_y_true, worst_subset_y_pred)


#####################
# quality functions #
#####################


def _label_balance_fraction(labels: pd.Series):
    if labels.nunique() != 2:
        return 0

    labels = labels.groupby(by=lambda x: labels[x]).count()

    if labels.iloc[1] == 0:
        return np.inf

    result = labels.iloc[0] / labels.iloc[1]

    if result > 1:
        result = 1 / result

    return result


class BaseSoftClassifierPerformanceQF(ps.BoundedInterestingnessMeasure):
    def __init__(
        self,
        performance_measure,
        performance_measure_type: Literal["score", "loss"],
        performance_measure_bound=None,
        performance_measure_constraints: list[any] = [],
        subgroup_class_balance_weight: float = 0,
        subgroup_size_weight: float = 0,
    ):
        """
        :param performance_measure: A function that maps lists of labels and predictions to a single float, measuring predictive performance.
        :param performance_measure_type: Determines whether higher is better ("score") or lower is better ("loss").
        :param performance_measure_bound: A function that returns a tight bound on the worst possible performance that performance_measure can return on any subset of the given labels and predictions. This is a lower bound for score-type performance measures and an upper bound for loss-type performance measures.
        :param performance_mesaure_constraints: A list of constraints that need to be fulfilled so that performance_measure is not undefined. In other words, these constraints describe the undefined cases of performance_measure.
        :param subgroup_class_balance_weight: Amplifies the quality score of subgroups with a more balanced class ratio.
        :param subgroup_size_weight: Amplifies the quality score of subgroups with a greater cover size.
        """

        self.performance_measure = performance_measure
        self.performance_measure_type = performance_measure_type
        self.performance_measure_bound = performance_measure_bound
        self.performance_measure_constraints = performance_measure_constraints

        self.subgroup_class_balance_weight = subgroup_class_balance_weight
        self.subgroup_size_weight = subgroup_size_weight

        # declare attributes for constant statistics
        self.scores_sorted = None
        self.gt_sorted_by_score = None
        self.sorted_to_original_index = None
        self.has_constant_statistics = None
        self.dataset_quality = None

    def calculate_constant_statistics(
        self, data: pd.DataFrame, target: SoftClassifierTarget
    ):
        """calculate_constant_statistics
        This function is called once for every search execution,
        it should do any preparation that is necessary prior to an execution.
        """
        dataset_sorted_by_score = data.sort_values(target.prediction_column)
        self.scores_sorted = dataset_sorted_by_score.loc[:, target.prediction_column]
        self.gt_sorted_by_score = dataset_sorted_by_score.loc[:, target.label_column]
        self.sorted_to_original_index = [
            index for index, _ in dataset_sorted_by_score.iterrows()
        ]

        y_true = self.gt_sorted_by_score.to_numpy()
        y_pred = self.scores_sorted.to_numpy()
        self.dataset_quality = self.performance_measure(y_true, y_pred)

        self.has_constant_statistics = True

    def calculate_statistics(
        self,
        subgroup,
        target: SoftClassifierTarget,
        data: pd.DataFrame,
        statistics=None,
    ):
        """calculates necessary statistics
        this statistics object is passed on to the evaluate
        and optimistic_estimate functions
        """
        if not hasattr(subgroup, "representation"):
            subgroup = ps.create_subgroup_with_representation(data, subgroup._selectors)

        return {"size_sg": sum(subgroup.representation)}

    def _get_quality_weight(self, subgroup, target, data: pd.DataFrame):
        subgroup_labels = data.loc[subgroup.representation, target.label_column]
        subgroup_label_balance_fraction = _label_balance_fraction(subgroup_labels)

        subgroup_size = len(subgroup_labels)

        return (
            subgroup_label_balance_fraction**self.subgroup_class_balance_weight
        ) * (subgroup_size**self.subgroup_size_weight)

    def evaluate(
        self,
        subgroup,
        target: SoftClassifierTarget,
        data: pd.DataFrame,
        statistics=None,
    ):
        """return the quality calculated from the statistics"""
        if subgroup == slice(None):
            sel_conjunction = ps.Conjunction.from_str("Dataset")
            subgroup = ps.create_subgroup_with_representation(
                data, sel_conjunction.selectors
            )

        if not hasattr(subgroup, "representation"):
            subgroup = ps.create_subgroup_with_representation(data, subgroup._selectors)

        if statistics is None:
            statistics = self.calculate_statistics(subgroup, target, data)

        if not ps.constraints_satisfied(
            self.performance_measure_constraints,
            subgroup,
            statistics,
            data,
        ):
            return (
                -np.inf
            )  # performance measure is undefined for subgroup -> maximally uninteresting

        sorted_subgroup_representation = [
            subgroup.representation[original_index]
            for original_index in self.sorted_to_original_index
        ]
        sorted_subgroup_y_true = self.gt_sorted_by_score[
            sorted_subgroup_representation
        ].to_numpy()
        sorted_subgroup_y_pred = self.scores_sorted[
            sorted_subgroup_representation
        ].to_numpy()
        statistics["performance measure"] = self.performance_measure(
            sorted_subgroup_y_true, sorted_subgroup_y_pred
        )

        quality = statistics["performance measure"] - self.dataset_quality

        if self.performance_measure_type == "score":
            quality = -quality

        return quality * self._get_quality_weight(
            subgroup, target, data, quality, statistics=statistics
        )

    def optimistic_estimate(
        self,
        subgroup,
        target: SoftClassifierTarget,
        data: pd.DataFrame,
        statistics=None,
    ):
        """returns optimistic estimate
        if one is available return it otherwise infinity"""
        # Stop if optimistic estimate is unknown.
        # Cannot estimate if no bound is given or quality weighting is used (at least one weight != 0) but the condition for
        # estimating the quality weight is not met.
        if (
            self.performance_measure_bound is None
            or 0 > self.subgroup_size_weight
            or self.subgroup_size_weight > self.subgroup_class_balance_weight
        ):
            return np.inf

        if not hasattr(subgroup, "representation"):
            subgroup = ps.create_subgroup_with_representation(data, subgroup._selectors)

        if statistics is None:
            statistics = self.calculate_statistics(subgroup, target, data, statistics)

        # step 1: prepare estimate input
        sorted_subgroup_representation = [
            subgroup.representation[original_index]
            for original_index in self.sorted_to_original_index
        ]
        sorted_subgroup_y_true = self.gt_sorted_by_score[
            sorted_subgroup_representation
        ].to_numpy()
        sorted_subgroup_y_pred = self.scores_sorted[
            sorted_subgroup_representation
        ].to_numpy()

        # step 2: compute estimate for most extreme performance measure result
        performance_measure_estimate = self.performance_measure_bound(
            sorted_subgroup_y_true, sorted_subgroup_y_pred
        )

        # step 3: postprocess the result as in evaluate()
        quality_estimate = performance_measure_estimate - self.dataset_quality

        if self.performance_measure_type == "score":
            quality_estimate = -quality_estimate

        # Add optimistic estimate of quality weighting in the special case where cover size and class balance parameter are both 1.
        # Otherwise both weights are 0 so no estimate is needed.
        if 0 < self.subgroup_size_weight <= self.subgroup_class_balance_weight:
            subgroup_labels = data.loc[subgroup.representation, target.label_column]

            if subgroup_labels.nunique() != 2:
                return 0  # class balance term in the quality weight is 0 and that cannot change for any refinement

            subgroup_labels = subgroup_labels.groupby(
                by=lambda x: subgroup_labels[x]
            ).count()
            min_label_count = min(subgroup_labels.iloc[0], subgroup_labels.iloc[1])
            quality_estimate *= (2 * min_label_count) ** self.subgroup_size_weight

        return quality_estimate


class ARLQF(BaseSoftClassifierPerformanceQF):
    def __init__(
        self,
        label_column: str,
        positive_label_value: any,
        subgroup_class_balance_weight: float = 0,
        subgroup_size_weight: float = 0,
    ):
        """
        :param label_column: column identifier of the labels / ground truth in the dataset
        :param positive_label_value: label value that is considered the positive class
        :param subgroup_class_balance_weight: amplifies the quality score of subgroups with a more balanced class ratio
        :param subgroup_size_weight: amplifies the quality score of subgroups with a greater cover size
        """

        # define a constraint in which case ARL is undefined
        constraints = [ps.ContainsValueConstraint(label_column, positive_label_value)]

        super().__init__(
            average_ranking_loss,
            "loss",
            _ARL_upper_bound,
            constraints,
            subgroup_class_balance_weight,
            subgroup_size_weight,
        )


class ROCAUCQF(BaseSoftClassifierPerformanceQF):
    def __init__(
        self,
        label_column: str,
        positive_label_value: any,
        negative_label_value: any,
        subgroup_class_balance_weight: float = 0,
        subgroup_size_weight: float = 0,
    ):
        """
        :param label_column: column identifier of the labels / ground truth in the dataset
        :param positive_label_value: label value that is considered the positive class
        :param negative_label_value: label value that is considered the negative class
        :param subgroup_class_balance_weight: amplifies the quality score of subgroups with a more balanced class ratio
        :param subgroup_size_weight: amplifies the quality score of subgroups with a greater cover size
        """

        # define constraints in which case ROC AUC is undefined
        constraints = [
            ps.ContainsValueConstraint(label_column, positive_label_value),
            ps.ContainsValueConstraint(label_column, negative_label_value),
        ]

        super().__init__(
            metrics.roc_auc_score,
            "score",
            _ROC_AUC_lower_bound,
            constraints,
            subgroup_class_balance_weight,
            subgroup_size_weight,
        )


class PRAUCQF(BaseSoftClassifierPerformanceQF):
    def __init__(
        self,
        label_column: str,
        positive_label_value: any,
        subgroup_class_balance_weight: float = 0,
        subgroup_size_weight: float = 0,
    ):
        """
        :param label_column: column identifier of the labels / ground truth in the dataset
        :param positive_label_value: label value that is considered the positive class
        :param subgroup_class_balance_weight: amplifies the quality score of subgroups with a more balanced class ratio
        :param subgroup_size_weight: amplifies the quality score of subgroups with a greater cover size
        """

        # define a constraint in which case PR AUC is undefined
        constraints = [ps.ContainsValueConstraint(label_column, positive_label_value)]

        super().__init__(
            pr_auc_score,
            "score",
            _PR_AUC_lower_bound,
            constraints,
            subgroup_class_balance_weight,
            subgroup_size_weight,
        )
