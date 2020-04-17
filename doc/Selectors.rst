##########
Selectors
##########

Selectors are objects that if applied to a dataset yield a set of instances. If an instance is retured from a selector we say that the selectors covers that instance.
While the term selectors usually only refers to basic selectors, conjunctions and disjunctions as well as negated selectors are also in a general sense selectors. Broadly speaking anything that implements the code:`covers` function is a selector.
We will first introduce the frequently used basic selectors and thereafter the more general selectors that are the conjunction and disjunction. We conclude the chapter by showing how to implement a selectors yourself.



Basic Selectors
################

The pysubgroup package provides two basic selectors: The EqualitySelector and the IntervalSelector.
Lets start by exploring the EqualitySelector:

.. highlight:: python

.. testcode::

    import pysubgroup as ps
    import pandas as pd

    # create dataset
    first_names = ['Alex', 'Anna', 'Alex']
    sur_names = ['Smith', 'Johnson', 'Williams']
    ages =  [40, 25, 32]
    df = pd.DataFrame.from_dict({'First_name':first_names, 'Sur_name': sur_names, 'age':ages})

    # create selector
    alex_selector = ps.EqualitySelector('First_name', 'Alex')
    age_selector = ps.EqualitySelector('age', 22)
    # apply selectors to dataframe
    print('instances with ', str(alex_selector), alex_selector.covers(df))
    print('instances with', str(age_selector), age_selector.covers(df))

.. testoutput::
    :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

    instances with  First_name=='Alex' [ True False  True]
    instances with age==22 [False False False]

The output indicates that the first and third instance in the dataset have a first name that is equal to :code:`'Alex'`.
The second output shows that no instances in our dataset is of age 22.
The EqualitySelector selector can be used on many different datatypes, but is most useful on binary, string and categorical data.
In addition to the EqualitySelector the pysubgroup package also provides the IntervalSelector. The following codes selects all instances from the database, which are in the age range 18 (included) to 40 (excluded).

.. testcode::

    interval_selector = ps.IntervalSelector('age', 18, 40)
    print(interval_selector.covers(df))

.. testoutput::

    [False  True  True]

The outpu shows that the second and third instance in our dataset have an age within the interval :math:`[18,40)`.

Selectors are the building block of all rules generated with the pysubgroup package. If you want to write your own custom selector that is not a problem see :ref:`customselector` for references.

Negations
################
The pysubgroup package also provides the NegatedSelector class that takes any selector (not just basic ones) and inverts it.

.. testcode::

    inverted_selector = ps.NegatedSelector(alex_selector)
    print('instances with first name not equal to Alex', inverted_selector.covers(df))

.. testoutput::

    instances with first name not equal to Alex [False  True False]

The output is: :code:`instances with first name not equal to Alex  [False, True, False]`.




Conjunctions
################
Most of the rules that are generated with the pysubgroup package use conjunctions to form more complex queries. Continuing the running example from above we can find all persons whose name is Alex *and* which have an age in the interval :math:`[18,40)` like so:

.. testcode::

    conj = ps.Conjunction([interval_selector, alex_selector])
    print('instances with', str(conj), conj.covers(df))

.. testoutput::

    instances with First_name=='Alex' AND age: [18:40[ [False False  True]

The output shows that only the last instance is covered by our conjunction.


Disjunctions
################

The pysubgroup package also provides disjunctions with the :code:`Disjunction` class. Continuing the running example we can find all persons whose name is Alex *or* which have an age in the interval :math:`[18,40)` like so:

.. testcode::

    disj = ps.Disjunction([interval_selector, alex_selector])
    print('instances with', str(disj), disj.covers(df))

.. testoutput::

    instances with First_name=='Alex' OR age: [18:40[ [ True  True  True]

We can see that all instances are covered by our conjunction.

.. _customselector

Implementing your own
###############################

As already mentioned in the introduction on selectors, anything that provides a cover function is a selector. In this case we will show how to implement a custom basic selector that checks whether a string contains a given substring:

.. testcode::

    class StrContainsSelector:
        def __init__(self, column, substr):
            self.column = column
            self.substr = substr

        def covers(self, df):
            return df[self.column].str.contains(self.substr).to_numpy()

    contains_selector = StrContainsSelector('Sur_name','m')
    print(contains_selector.covers(df))

.. testoutput::

    [ True False  True]

The output shows that only the first and last instance contain an m in their name.
In addition to the covers function it is certainly advised to also implement the :code:`__str__` and :code:`__repr__` functions. This selector can now be added to the searchspace for any algorithm execution.
    