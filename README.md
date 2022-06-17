# Pysubgroup

**pysubgroup** is a Python package that enables subgroup discovery in Python+pandas (scipy stack) data analysis environment. It provides for a lightweight, easy-to-use, extensible and freely available implementation of state-of-the-art algorithms, interestingness measures and presentation options.

As of 2020, this library is still in a prototype phase. It has, however, been already succeesfully employed in active application projects.

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

* Herrera, Franciso, et al. ["An overview on subgroup discovery: foundations and applications."](https://scholar.google.de/scholar?q=Herrera%2C+Franciso%2C+et+al.+%E2%80%9CAn+overview+on+subgroup+discovery%3A+foundations+and+applications.%E2%80%9D+Knowledge+and+information+systems+29.3+(2011)%3A+495-525.) Knowledge and information systems 29.3 (2011): 495-525.
* Atzmueller, Martin. ["Subgroup discovery."](https://scholar.google.de/scholar?q=Atzmueller%2C+Martin.+%E2%80%9CSubgroup+discovery.%E2%80%9D+Wiley+Interdisciplinary+Reviews%3A+Data+Mining+and+Knowledge+Discovery+5.1+(2015)%3A+35-49.) Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery 5.1 (2015): 35-49.
* And of course, my point of view on the topic is [summarized in my dissertation](https://opus.bibliothek.uni-wuerzburg.de/files/9781/Dissertation-Lemmerich.pdf):

### Prerequisites and Installation
pysubgroup is built to fit in the standard Python data analysis environment from the scipy-stack.
Thus, it can be used just having pandas (including its dependencies numpy, scipy, and matplotlib) installed. Visualizations are carried out with the matplotlib library.

pysubgroup consists of pure Python code. Thus, you can simply download the code from the repository and copy it in your `site-packages` directory.
pysubgroup is also on PyPI and should be installable using:  
`pip install pysubgroup`

**Note**: Some users complained about the **pip installation not working**.
If, after the installation, it still doesn't find the package, then do the following steps:
 1. Find where the directory `site-packages` is.
 2. Copy the folder `pysubgroup`, which contains the source code, into the `site-packages` directory. (WARNING: This is not the main repository folder. The `pysubgroup` folder is inside the main repository folder, at the same level as `doc`)
 3. Now you can import the module with `import pysubgroup`.

### How to use:
A simple use case (here using the well known _titanic_ data) can be created in just a few lines of code:

```python
import pysubgroup as ps

# Load the example dataset
from pysubgroup.tests.DataSets import get_titanic_data
data = get_titanic_data()

target = ps.BinaryTarget ('Survived', True)
searchspace = ps.create_selectors(data, ignore=['Survived'])
task = ps.SubgroupDiscoveryTask (
    data, 
    target, 
    searchspace, 
    result_set_size=5, 
    depth=2, 
    qf=ps.WRAccQF())
result = ps.BeamSearch().execute(task)
```
The first line imports _pysubgroup_ package.
The following lines load an example dataset (the popular titanic dataset).

Therafter, we define a target, i.e., the property we are mainly interested in (_'survived'}.
Then, we define the searchspace as a list of basic selectors. Descriptions are built from this searchspace. We can create this list manually, or use an utility function.
Next, we create a SubgroupDiscoveryTask object that encapsulates what we want to find in our search.
In particular, that comprises the target, the search space, the depth of the search (maximum numbers of selectors combined in a subgroup description), and the interestingness measure for candidate scoring (here, the Weighted Relative Accuracy measure).

The last line executes the defined task by performing a search with an algorithm---in this case beam search. The result of this algorithm execution is stored in a SubgroupDiscoveryResults object.

To just print the result, we could for example do:

```python
print(result.to_dataframe())
```

to get:

<table border="1" class="dataframe">
<thead>    <tr style="text-align: right;">      <th></th>      <th>quality</th>      <th>description</th>    </tr>  </thead>
<tbody>
    <tr>      <th>0</th>      <td>0.132150</td>      <td>Sex==female</td>    </tr>
    <tr>      <th>1</th>      <td>0.101331</td>      <td>Parch==0 AND Sex==female</td>    </tr>
    <tr>      <th>2</th>      <td>0.079142</td>      <td>Sex==female AND SibSp: [0:1[</td>    </tr>
    <tr>      <th>3</th>      <td>0.077663</td>      <td>Cabin.isnull() AND Sex==female</td>    </tr>
    <tr>      <th>4</th>      <td>0.071746</td>      <td>Embarked==S AND Sex==female</td>    </tr>
</tbody></table>


### Key classes
Here is an outline on the most important classes:
* Selector: A Selector represents an atomic condition over the data, e.g., _age < 50_. There several subtypes of Selectors, i.e., NominalSelector (color==BLUE), NumericSelector (age < 50) and NegatedSelector (a wrapper such as not selector1)
* SubgroupDiscoveryTask: As mentioned before, encapsulates the specification of how an algorithm should search for interesting subgroups
* SubgroupDicoveryResult: These are the main outcome of a subgroup disovery run. You can obtain a list of subgroups using the `to_subgroups()` or to a dataframe using `to_dataframe()`
* Conjunction: A conjunction is the most widely used SubgroupDescription, and indicates which data instances are covered by the subgroup. It can be seen as the left hand side of a rule.





### License
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
    
### Cite
If you are using pysubgroup for your research, please consider citing our demo paper:
        
    Lemmerich, F., & Becker, M. (2018, September). pysubgroup: Easy-to-use subgroup discovery in python. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases (ECMLPKDD). pp. 658-662.
    
bibtex:
  
    @inproceedings{lemmerich2018pysubgroup,
      title={pysubgroup: Easy-to-use subgroup discovery in python},
      author={Lemmerich, Florian and Becker, Martin},
      booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
      pages={658--662},
      year={2018}
    }
