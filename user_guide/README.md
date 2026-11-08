# Subgroup Discovery with pysubgroup — book sources

This Jupyter Book 2 user guide contains twenty chapters, five reference pages,
and an introductory landing page. Read `index.md` for the reader's entry point.

## Contents and sources

- `01-...md` through `20-...md`: the six parts of the book. Chapters 2–20 are
  executable MyST Markdown with Python kernelspec metadata.
- `reference-*.md`: method selection, API quick reference, glossary,
  troubleshooting/migration, and bibliography.
- `references.bib`: shared publication citations.
- `myst.yml`: native Jupyter Book 2 configuration and explicit navigation.
- `build.py`: build/preview helper with book-local caches and runtime paths.
- `one-kernel.cjs`: process-local concurrency workaround for the pinned CLI.
- `package.json`, `package-lock.json`: pinned Jupyter Book 2.1.7 toolchain.
- `requirements-book.txt`: Python authoring dependencies, in addition to the
  package's own dependencies.

Each executable chapter is self-contained. Most examples use offline
scikit-learn loaders or explicitly generated data. Chapters 2 and 15 download
Adult from OpenML, share a local cache, and analyze the same deterministic
3,000-row sample.
No datasets are committed with the book.

## Authoring environment

Use Python 3.12 or newer within the package's supported range, with the
dependencies of `pysubgroup[plot]`. In a normal authoring virtual environment,
also install the packages listed in `requirements-book.txt`.

The build helper reads this checkout's `../src` directly. It does not install or
edit the package, change the repository's dependency files, or run pytest.

Install the book-local JavaScript CLI from this folder:

```console
npm ci --ignore-scripts --no-bin-links --cache .cache/npm
```

Node.js 18 or newer is required. If Python tools are installed in
`user_guide/.tools/python` with pip's `--target` option, the helper detects them
automatically.

## Check, build, and preview

Run **from this folder**, using the Python executable of your authoring
environment:

```console
python3 -B build.py check
python3 -B build.py html
python3 -B build.py preview
```

- `check` executes changed chapters and performs a strict content build.
- `html` additionally exports the static website to `_build/html/`.
- `preview` starts the development website and prints its local URL.

Execution outputs are cached by MyST under `_build/execute/`; unchanged chapters
reuse them. Add `--no-execute` to render using existing output caches. To
re-execute everything, remove only the generated `_build/execute/` cache before
building again. Changes to the package itself are not part of MyST's cell-source
cache key, so invalidate that cache after a relevant package change.

The helper routes home/configuration paths, temporary files, Jupyter runtime
files, plotting caches, numba caches, and tool caches into this folder. It
registers a local `python3` kernelspec pointing to the executable running it.
On FUSE file systems that do not preserve requested runtime-file modes, Jupyter
needs its `JUPYTER_ALLOW_INSECURE_WRITES` compatibility setting. The generated
server credentials stay in the ignored local cache.

Jupyter Book 2.1.7 accepts `--execute-parallel` but its internal semaphore still
uses CPU count. `one-kernel.cjs` limits the CPU count visible to the CLI process
so this book executes at most one chapter kernel at a time. It changes no
installed tool sources or system CPU settings.

For offline viewing of the static output, serve it over HTTP:

```console
python3 -B -m http.server --directory _build/html 8000
```

Then open `http://localhost:8000`. Use the development preview for live authoring.

## Conventions

- Keep all sources, tooling, figures, caches, data downloads, and generated
  output inside `user_guide/`.
- Write executable MyST Markdown as the single source for explanations and
  code. Use notebooks only when their editing workflow adds value.
- Start each executable chapter with explicit imports, data loading, and seeds.
- Prefer small, shallow searches and keep expensive benchmarks out of builds.
- Explain the question, reference group, output, and interpretation alongside
  the code. Cite the publications defining methods and measures.
- Check the current public API when adapting existing documentation. Keep
  positional-mask semantics and `(lower, upper]` interval boundaries explicit.

## Tool documentation

- [Jupyter Book 2](https://jupyterbook.org/stable/)
- [MyST table of contents](https://mystmd.org/guide/table-of-contents)
- [Executable Markdown](https://mystmd.org/guide/notebooks-with-markdown)
- [Notebook execution](https://mystmd.org/guide/execute-notebooks)
