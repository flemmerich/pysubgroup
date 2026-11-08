"""Subgroup discovery search algorithms.

``base_algorithms`` holds ``SubgroupDiscoveryTask`` and the searches built on the
shared ``_Search`` (DFS, best-first, OPUS, DpSubgroup, Apriori, SD-Map, SD-Map*,
VLSD, beam search); ``numbsd`` holds NumBSD for numeric targets, ``dssd`` the
diverse subgroup set discovery DSSD, and ``generalization_aware_algorithms``
the searches for generalization-aware measures.
"""

from .base_algorithms import (
    DFS,
    OPUS,
    VLSD,
    Apriori,
    BeamSearch,
    BestFirstSearch,
    DpSubgroup,
    SDMap,
    SDMapStar,
    SubgroupDiscoveryTask,
)
from .dssd import DSSD, cover_redundancy, joint_entropy, subgroup_covers
from .generalization_aware_algorithms import GeneralizationAwareApriori, GeneralizationAwareDFS
from .numbsd import NumBSD

__all__ = [
    "DFS",
    "DSSD",
    "OPUS",
    "VLSD",
    "Apriori",
    "BeamSearch",
    "BestFirstSearch",
    "DpSubgroup",
    "GeneralizationAwareApriori",
    "GeneralizationAwareDFS",
    "NumBSD",
    "SDMap",
    "SDMapStar",
    "SubgroupDiscoveryTask",
    "cover_redundancy",
    "joint_entropy",
    "subgroup_covers",
]
