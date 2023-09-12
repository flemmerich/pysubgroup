# Contributing

```
TODO: UPDATE THIS
```

Welcome to `pysubgroup` contributor's guide.

This document focuses on getting any potential contributor familiarized with
the development processes, but [other kinds of contributions] are also appreciated.

If you are new to using [git] or have never collaborated in a project previously,
please have a look at [contribution-guide.org]. Other resources are also
listed in the excellent [guide created by FreeCodeCamp] [^contrib1].

Please notice, all users and contributors are expected to be **open,
considerate, reasonable, and respectful**. When in doubt,
[Python Software Foundation's Code of Conduct] is a good reference in terms of
behavior guidelines.

## Issue Reports

If you experience bugs or general issues with `pysubgroup`, please have a look
on the [issue tracker].
If you don't see anything useful there, please feel free to fire an issue report.

:::{tip}
Please don't forget to include the closed issues in your search.
Sometimes a solution was already reported, and the problem is considered
**solved**.
:::

New issue reports should include information about your programming environment
(e.g., operating system, Python version) and steps to reproduce the problem.
Please try also to simplify the reproduction steps to a very minimal example
that still illustrates the problem you are facing. By removing other factors,
you help us to identify the root cause of the issue.

## Documentation Improvements

You can help improve `pysubgroup` docs by making them more readable and coherent, or
by adding missing information and correcting mistakes.

`pysubgroup` documentation uses [Sphinx] as its main documentation compiler.
This means that the docs are kept in the same repository as the project code, and
that any documentation update is done in the same way as a code contribution.
We are using [CommonMark] with [MyST] extensions as our markup language.

:::{tip}
   Please notice that the [GitHub web interface] provides a quick way of
   propose changes in `pysubgroup`'s files. While this mechanism can
   be tricky for normal code contributions, it works perfectly fine for
   contributing to the docs, and can be quite handy.

   If you are interested in trying this method out, please navigate to
   the `docs` folder in the source [repository], find which file you
   would like to propose changes and click in the little pencil icon at the
   top, to open [GitHub's code editor]. Once you finish editing the file,
   please write a message in the form at the bottom of the page describing
   which changes have you made and what are the motivations behind them and
   submit your proposal.
:::

When working on documentation changes in your local machine, you can
compile them using [tox] :

```
tox -e docs
```

and use Python's built-in web server for a preview in your web browser
(`http://localhost:8000`):

```
python3 -m http.server --directory 'docs/_build/html'
```

## Code Contributions

```{todo} Please include a reference or explanation about the internals of the project.
   TODO: An architecture description, design principles or at least a summary of the
   main concepts will make it easy for potential contributors to get started
   quickly.
```

### Submit an issue

Before you work on any non-trivial code contribution it's best to first create
a report in the [issue tracker] to start a discussion on the subject.
This often provides additional considerations and avoids unnecessary work.

### Create an environment

Before you start coding, we recommend creating an isolated [virtual environment]
to avoid any problems with your installed Python packages.
This can easily be done via either [virtualenv]:

```
virtualenv <PATH TO VENV>
source <PATH TO VENV>/bin/activate
```

or [Miniconda]:

```
conda create -n pysubgroup python=3 six virtualenv pytest pytest-cov
conda activate pysubgroup
```

### Clone the repository

1. Create an user account on GitHub if you do not already have one.

2. Fork the project [repository]: click on the *Fork* button near the top of the
   page. This creates a copy of the code under your account on GitHub.

3. Clone this copy to your local disk:

   ```
   git clone git@github.com:YourLogin/pysubgroup.git
   cd pysubgroup
   ```

4. You should run:

   ```
   pip install -U pip setuptools -e .
   ```

   to be able to import the package under development in the Python REPL.

   ```{todo} if you are not using pre-commit, please remove the following item:
   ```

5. Install [pre-commit]:

   ```
   pip install pre-commit
   pre-commit install
   ```

   `pysubgroup` comes with a lot of hooks configured to automatically help the
   developer to check the code being written.

### Implement your changes

1. Create a branch to hold your changes:

   ```
   git checkout -b my-feature
   ```

   and start making changes. Never work on the main branch!

2. Start your work on this branch. Don't forget to add [docstrings] to new
   functions, modules and classes, especially if they are part of public APIs.

3. Add yourself to the list of contributors in `AUTHORS.rst`.

4. When you’re done editing, do:

   ```
   git add <MODIFIED FILES>
   git commit
   ```

   to record your changes in [git].

   ```{todo} if you are not using pre-commit, please remove the following item:
   ```

   Please make sure to see the validation messages from [pre-commit] and fix
   any eventual issues.
   This should automatically use [flake8]/[black] to check/fix the code style
   in a way that is compatible with the project.

   :::{important}
   Don't forget to add unit tests and documentation in case your
   contribution adds an additional feature and is not just a bugfix.

   Moreover, writing a [descriptive commit message] is highly recommended.
   In case of doubt, you can check the commit history with:

   ```
   git log --graph --decorate --pretty=oneline --abbrev-commit --all
   ```

   to look for recurring communication patterns.
   :::

5. Please check that your changes don't break any unit tests with:

   ```
   tox
   ```

   (after having installed [tox] with `pip install tox` or `pipx`).

   You can also use [tox] to run several other pre-configured tasks in the
   repository. Try `tox -av` to see a list of the available checks.

### Submit your contribution

1. If everything works fine, push your local branch to the remote server with:

   ```
   git push -u origin my-feature
   ```

2. Go to the web page of your fork and click "Create pull request"
   to send your changes for review.

   ```{todo} if you are using GitHub, you can uncomment the following paragraph

      Find more detailed information in [creating a PR]. You might also want to open the PR as a draft first and mark it as ready for review after the feedbacks from the continuous integration (CI) system or any required fixes.

   ```

### Troubleshooting

The following tips can be used when facing problems to build or test the
package:

1. Make sure to fetch all the tags from the upstream [repository].
   The command `git describe --abbrev=0 --tags` should return the version you
   are expecting. If you are trying to run CI scripts in a fork repository,
   make sure to push all the tags.
   You can also try to remove all the egg files or the complete egg folder, i.e.,
   `.eggs`, as well as the `*.egg-info` folders in the `src` folder or
   potentially in the root of your project.

2. Sometimes [tox] misses out when new dependencies are added, especially to
   `setup.cfg` and `docs/requirements.txt`. If you find any problems with
   missing dependencies when running a command with [tox], try to recreate the
   `tox` environment using the `-r` flag. For example, instead of:

   ```
   tox -e docs
   ```

   Try running:

   ```
   tox -r -e docs
   ```

3. Make sure to have a reliable [tox] installation that uses the correct
   Python version (e.g., 3.7+). When in doubt you can run:

   ```
   tox --version
   # OR
   which tox
   ```

   If you have trouble and are seeing weird errors upon running [tox], you can
   also try to create a dedicated [virtual environment] with a [tox] binary
   freshly installed. For example:

   ```
   virtualenv .venv
   source .venv/bin/activate
   .venv/bin/pip install tox
   .venv/bin/tox -e all
   ```

4. [Pytest can drop you] in an interactive session in the case an error occurs.
   In order to do that you need to pass a `--pdb` option (for example by
   running `tox -- -k <NAME OF THE FALLING TEST> --pdb`).
   You can also setup breakpoints manually instead of using the `--pdb` option.

## Maintainer tasks

### Releases

If you are part of the group of maintainers and have correct user permissions
on [PyPI] and [Conda-Forge], the following steps can be used to release a new version for
`pysubgroup`:

#### PyPI

1. Merge `master` branch into `develop`.
2. Make sure all unit tests are successful on `develop`.
3. Create a pull request from the `develop` to the `master` branch.
4. Merge the pull request after all tests have passed.
5. Tag the current commit on the main branch (`master`)
      with a release tag, e.g., `git tag -a 0.7.7 -m "Release 0.7.7`.
6. Push the new tag to the upstream,
   e.g., `git push --tags`
7. GitHub Actions will now automatically push this version to
7. Clean up the `dist` and `build` folders with `tox -e clean`
   (or `rm -rf dist build`)
   to avoid confusion with old builds and Sphinx docs.
8. The

#### Conda Forge

  TODO: Automate this with GitHub Actions?

Resources:

- Conda: [Contributing packages](https://conda-forge.org/docs/maintainer/adding_pkgs.html)


1. **Fork** and clone [pysubgroup-feedstock](https://github.com/conda-forge/pysubgroup-feedstock)
2. Create a branch with the current version number,
   e.g., `git branch 0.7.7; git checkout 0.7.7`.
3. Get the `sha256` hash [from PyPI](https://pypi.org/project/pysubgroup),
   i.e., click on `view hashes` of the `Source Distribution`.
4. Update the `recipe/meta.yaml`:
```yaml
{% set version = "0.7.7" %}

# ...

source:
  sha256: <SHA256 from PyPI>
```
5. Commit and push branch
6. Create a pull request on main repository.
   For this, make sure to fill in all the fields
   and use [closing tags](https://docs.github.com/en/issues/tracking-your-work-with-issues) for changes.
7. Merge the pull requests when all tests have successfully passed.


[^contrib1]: Even though, these resources focus on open source projects and
    communities, the general ideas behind collaborating with other developers
    to collectively create software are general and can be applied to all sorts
    of environments, including private companies and proprietary code bases.

[black]: https://pypi.org/project/black/
[commonmark]: https://commonmark.org/
[contribution-guide.org]: http://www.contribution-guide.org/
[creating a pr]: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request
[descriptive commit message]: https://chris.beams.io/posts/git-commit
[docstrings]: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
[first-contributions tutorial]: https://github.com/firstcontributions/first-contributions
[flake8]: https://flake8.pycqa.org/en/stable/
[git]: https://git-scm.com
[github web interface]: https://docs.github.com/en/github/managing-files-in-a-repository/managing-files-on-github/editing-files-in-your-repository
[github's code editor]: https://docs.github.com/en/github/managing-files-in-a-repository/managing-files-on-github/editing-files-in-your-repository
[github's fork and pull request workflow]: https://guides.github.com/activities/forking/
[guide created by freecodecamp]: https://github.com/freecodecamp/how-to-contribute-to-open-source
[miniconda]: https://docs.conda.io/en/latest/miniconda.html
[myst]: https://myst-parser.readthedocs.io/en/latest/syntax/syntax.html
[other kinds of contributions]: https://opensource.guide/how-to-contribute
[pre-commit]: https://pre-commit.com/
[pypi]: https://pypi.org/
[pyscaffold's contributor's guide]: https://pyscaffold.org/en/stable/contributing.html
[pytest can drop you]: https://docs.pytest.org/en/stable/usage.html#dropping-to-pdb-python-debugger-at-the-start-of-a-test
[python software foundation's code of conduct]: https://www.python.org/psf/conduct/
[restructuredtext]: https://www.sphinx-doc.org/en/master/usage/restructuredtext/
[sphinx]: https://www.sphinx-doc.org/en/master/
[tox]: https://tox.readthedocs.io/en/stable/
[virtual environment]: https://realpython.com/python-virtual-environments-a-primer/
[virtualenv]: https://virtualenv.pypa.io/en/stable/


```{todo} Please review and change the following definitions:
```

[repository]: https://github.com/<USERNAME>/pysubgroup
[issue tracker]: https://github.com/<USERNAME>/pysubgroup/issues
