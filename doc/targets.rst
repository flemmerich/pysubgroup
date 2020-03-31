##############################
Targets and Quality Functions
##############################

To define the goal of our subgroup discovery task, we use targets and quality functions. Targets are used to define which attributes play a significant role and can provide common statistics for a subgroup in question. Quality functions assign a score to each subgroup.
These scores are used by all the algorithms to determine the most interesting subgroups. 

Frequency Targets
##################

The most simple target is the *FITarget* with its associated quality functions *CountQF* and *AreaQf*.
The CountQF simple counts the number of instances covered by the subgroup in question.
The AreaQF multiplies the depth or length of the subgroup description with the number of instances covered by that description.

Binary Targets
##################

For Boolean or Binary Targets we provide the *ChiSquaredQF* as well as the *StandardQF* quality functions.
The *StandardQF* quality function uses a parameter :math:`\alpha` to weight the relative size :math:`\frac{N_{SG}}{N}` of a subgroup and 
multiplies it with the differences in relations of positive instances :math:`p` to the number of instances :math:`N`

.. math::

    \left ( \frac{N_{SG}}{N} \right ) ^\alpha \left(\frac{p_{SG}}{N_{SG}} - \frac{p}{N} \right)

The *StandardQF* also supports an optimistic estimate. 

.. TODO

    optimistic estimate


The *ChiSquaredQF* is calculated based on the following contigency table which is then passed to the scipy `chi2_contigency <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html>`_ function.
The small :math:`n` represents the number of negative instances and should not be confused with the capital :math:`N` which represents the total number of instances.

+----------------+-----------------+
| :math:`p_{SG}` | :math:`p-p_{SG}`|
+----------------+-----------------+
| :math:`n_{SG}` | :math:`n-n_{SG}`|
+----------------+-----------------+



Numeric Targets
##################