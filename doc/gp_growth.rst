##########
GP-Growth
##########


This tree based algorithm uses a condensed representation (a so called valuation basis) to find interesting subgroups. The main advantage of this approach is, that the (potentially large) database has to be scanned only twice and thereafter all the necessary information is represented as more compact pattern-tree.
Gp-growth is a generalisation of the popular `fp-growth <https://en.wikipedia.org/wiki/Association_rule_learning#FP-growth_algorithm>`_ algorithm. So refer to instructional material on fp-growth for more in depth knowledge on the workings of this tree based algorithm.


.. contents:: 
    :depth: 2

Basic usage
########################

The basic usage of the gp-growth algorithm is not very different from the usage of any other algorithm in this package.

.. code-block:: python

    import pysubgroup as ps

    # Load the example dataset
    from pysubgroup.tests.DataSets import get_titanic_data
    data = get_titanic_data()

    target = ps.NominalSelector ('Survived', True)
    searchspace = ps.create_selectors(data, ignore=['Survived'])
    task = ps.SubgroupDiscoveryTask (data, target, dearchspace, result_set_size=5, depth=2, qf=ps.WRAccQF())
    GpGrowth.execute(task)

But beware that gp-growth is using an exhaustive search strategy! This can greatly increase the runtime for high search depth.
You can specify the :code:`mode` argument in the constructor of GpGrowth to run gp-growth either bottom up (:code:`mode='b_u'`) or top down (:code:`mode='b_u'`).
As gp growth is a generalisation of fp-growth you can also perform standard fp-growth using gp_growth by using the CountQF (:ref:`countqf`) quality function.


.. 
    Export a gp_tree
    =================
    It is possible to export a gp_tree. 



Create a custom target
##############################

If you consider to use the gp-growth algorithm for your custom target that is totally possible if you find a valuation basis.
We will now first introduce the concept of a valuation basis and thereafter outline the gp-growth interface that you have to support to use your quality function with our gp-growth implementation.

Valuation Basis
=================
Think of a valuation basis as a codensed representation of a subgroup that allows to quickly compute the same representation for a union of two disjoint subgroups.

We call the function which takes the valuation basis of two disjoint sets and computes the valuation basis for the unified set :code:`merge`. The function that compute the necessary statistics from a valuation basis :code:`stats_from_basis`.

Now we can formulate: Given two disjoint sets :math:`A` and :math:`B` with :math:`A \cap B = \varnothing` and their valuation bases :math:`v(A)` and :math:`v(B)` with their functions :code:`stats_from_basis` and :code:`merge` as defined above, we can compute the properties of :math:`A \cup B` instead of from the union of the instances from the merged valuation basis.
This can be summarized through the following equation:

.. math::

    props\_from\_instances(A\cup B) = props\_from\_basis(merge(v(A), v(B)))



Required Methods
=================
To make a target and quality function suitable for gp-growth you have to provide several methods (all methods start with :code:`gp_` to indicate that they are used in the gp-growth algorithm). In addition to the standard quality function methods (see :ref:`customtarget`) the following methods should be implemented to make a quality funciton usable with gp_growth.

.. code:: python

    class MyGpQualityFunction
        def gp_get_basis(self, row_index):
        """ returns the valuation basis of the element at this row_index """
            pass
        
        def gp_get_null_vector(self):
        """ returns the zero element of the valuation basis """
            pass

        @staticmethod
        def gp_merge(v_l, v_r):
        """ merges the v_r valuation basis into the v_l valuation basis inplace! """
            pass

        def gp_get_statistics(self, cover_arr, v):
        """ computes the statistics for this quality function from the valuation basis v """
            pass
        
        @property
        def gp_requires_cover_arr(self) -> bool:
        """ returns a boolean value that indicates whether a cover array is required when calling the gp_get_statistics function 

            usually this value is False
        """
            pass



Saving a gp_tree
=================

It is possible to save a gp tree to a txt file for e.g. debugging purpose. You therefor have to implementd the gp_to_str function which takes a valuation basis and returns a string representation.
It is an intentional choide to not call the  :code:`str` function on the valuation basis directly.

.. code:: python

    def gp_to_str(self, basis) -> str:
    """ returns a string representation of the valuation basis """
        pass