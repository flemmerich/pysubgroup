# Pysubgroup

**pysubgroup** is a Python package that enables subgroup discovery in Python+pandas (scipy stack) data analysis environment. It provides for a lightweight, easy-to-use, extensible and freely available implementation of state-of-the-art algorithms, interestingness measures and presentation options.

As of 2018, this library is still in a prototype phase. It has, however, been already succeesfully employed in active application projects.

### Subgroup Discovery

Subgroup Discovery is a well established data mining technique that allows you to identify patterns in your data.
More precisely, the goal of subgroup discovery is to identify descriptions of data subsets that show an interesting distribution with respect to a pre-specified target concept.
For example, given a dataset of patients in a hospital, we could be interested in subgroups of patients, for which a certain treatment X was successful.
One example result could then be stated as:

_"While in general the operation is successful in only 60% of the cases", for the subgroup
of female patients under 50 that also have been treated with drug d, the successrate was 82%."_

Here, a variable _operation success_ is the target concept, the identified subgroup has the interpretable description _female=True AND age<50 AND drug_D = True_. We call these single conditions (such as _female=True_) selection expressions or short _selectors_.
The interesting behavior for this subgroup is that the distribution of the target concept differs significantly from the distribution in the overall general dataset.
A discovered subgroup could also be seen as a rule:
```
female=True AND age<50 AND drug_D = True ==> Operation_outcome=SUCCESS
```
Computationally, subgroup discovery is challenging since a large number of such conjunctive subgroup descriptions have to be considered. Of course, finding computable criteria, which subgroups are likely interesting to a user is also an eternal struggle. 
Therefore, a lot of literature has been devoted to the topic of subgroup discovery (including some of my own work). Recent overviews on the topic are for example:

* Herrera, Franciso, et al. "An overview on subgroup discovery: foundations and applications." Knowledge and information systems 29.3 (2011): 495-525.
* Atzmueller, Martin. "Subgroup discovery." Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery 5.1 (2015): 35-49.
* And of course, my point of view on the topic is [summarized in my dissertation](https://opus.bibliothek.uni-wuerzburg.de/files/9781/Dissertation-Lemmerich.pdf):

### Prerequisites and Installation
pysubgroup is built to fit in the standard Python data analysis environment from the scipy-stack.
Thus, it can be used just having pandas (including its dependencies numpy, scipy, and matplotlib) installed. Visualizations are carried out with the matplotlib library.

pysubgroup consists of pure Python code. Thus, you can simply download the code from the repository and copy it in your `site-packages` directory.
pysubgroup is also on PyPI and should be installable using:  
`pip install pysubgroup`

### How to use:
A simple use case (here using the well known _titanic_ data) can be created in just a few lines of code:

```python
import pysubgroup as ps
import pandas as pd

data = pd.read_csv("C:/data/titanic.csv")
target = ps.NominalTarget ('survived', True)
searchspace = ps.create_selectors(data, ignore=['survived'])
task = ps.SubgroupDiscoveryTask (data, target, searchspace, 
            result_set_size=5, depth=2, qf=ps.ChiSquaredQF())
result = ps.BeamSearch().execute(task)
```
he first two lines import the _pandas_ data analysis environment and the _pysubgroup_ package.
The following line loads the data into a standard pandas DataFrame object. The next three lines specify a subgroup discovery task. 
First, we define a target, i.e., the property we are mainly interested in (_'survived'}.
Then, we build a list of basic selectors to build descriptions from. We can create this list manually, or use an utility function.
Next, we create a SubgroupDiscoveryTask object that encapsulates what we want to find in our search.
In particular, that comprises the target, the search space, the depth of the search (maximum numbers of selectors combined in a subgroup description), and the interestingness measure for candidate scoring (here, the $\chi^2$ measure).

The last line executes the defined task by performing a search with an algorithm---in this case beam search. The result is then stored in a list of discovered subgroups associated with their score according to the chosen interestingness measure.

As a result, we obtain a list of tuples (score, subgroup_object):
To just print the result, we could for example do:

```python
for (q, sg) in result:
    print (str(q) + ":\t" + str(sg.subgroup_description)
```

to get:
```python
52.30879551820728:	Sex=female
52.30879551820728:	Sex=male
36.528775497473646:	Sex=female AND Parch=0
32.07982038616973:	Sex=female AND SibSp: [0:1[
31.587048503611975:	Sex=male AND Parch=0
```
However, there are also utility functions that allow you to export results to a pandas dataframe, to csv, or directly to latex.

### Key classes
Admittedly, the documentation of pysubgroup is currently lacking :(
Anyways, here is an outline on the most important classes:
* Subgroup: subgroup objects are the key outcome of any subgroup discovery algorithm. They encapsulate the target of the search, a _SubgroupDescription_, and statistics (size, target proportion, etc...)
* SubgroupDescription: subgroup descriptions specify, which data instances are covered by the subgroup. It can be seen as the left hand side of a rule. A SubgroupDescription stores a list of _Selectors_, which are interpreted as a conjunction.
* Selector: A Selector represents an atomic condition over the data, e.g., _age < 50_. There several subtypes of Selectors, i.e., NominalSelector (color==BLUE), NumericSelector (age < 50) and NegatedSelector (a wrapper such as not selector1)
* SubgroupDiscoveryTask: As mentioned before, encapsulates the specification of how an algorithm should search for interesting subgroups



**License**
I am happy about anyone using this software. Thus, this work is put under an Apache license. However, if this constitutes
any hindrance to your application, please feel free to contact me, I am sure that we can work something out.

    Copyright 2016-2019 Florian Lemmerich
        
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
