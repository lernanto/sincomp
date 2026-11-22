# AGENTS.md

## Scope

This file applies to the SinComp repository root and all its subdirectories. Unless a more specific `AGENTS.md` exists in a nested directory, these requirements apply to all source code, tests, scripts, and documentation under that directory.

## Project overview

SinComp is a Python toolkit for processing and comparing Chinese dialect pronunciation data. It supports loading, cleaning, and normalizing dialect datasets; aligning character pronunciations; calculating similarities; comparing rule compliance; statistical plotting; and model training.

The main repository structure is:

```text
sincomp/
├── .github/          GitHub configuration and workflows
├── docs/             User and API guides
├── src/
│   └── sincomp/      Core Python package, public API, and CLI
│       ├── _internal/ Internal algorithm implementations
│       └── cli/      Human- and agent-oriented CLI adapters
├── tests/            Automated tests and test data
├── pyproject.toml    Project metadata and build configuration
└── README.md         Project overview and quick start
```

Data, experiment results, and notebooks are not part of the core library interface. Keep changes to them narrowly scoped.

The package CLI provides human-friendly top-level subcommands through `sincomp`, and stable agent-oriented entry points through `sincomp.cli.*`. CLI adapters handle argument parsing, output formatting, and calls to the business APIs; algorithm implementations belong in the relevant `sincomp` modules. Output formats such as CSV files are part of the business interface and must not change merely to adjust terminal output.

Datasets may be downloaded lazily and cached locally. Tests and development code must not assume remote data is always available; prefer the smallest repository test data or fixed inputs.

## Documentation

Documentation is maintained in the `docs/` directory. Developers should at least read `docs/concepts.md` to understand the project's terminology and fundamental concepts.

Consult `docs/index.md` and the relevant topic guides for workflows, data conventions, and API usage. For command-line tasks, see `docs/cli.md` and the command's built-in help.

## Coding standards

### General principles

- Fix root causes and keep changes focused; avoid unrelated refactoring.
- Follow existing module boundaries, public APIs, and naming conventions. Add abstractions only when they reduce real complexity or match an established pattern.
- Do not modify generated data, notebooks, caches, model files, or other experimental artifacts unless explicitly requested.
- Avoid comments that do not aid understanding. Add a brief English explanation before complex logic when needed.
- Default to ASCII; use non-ASCII characters only when the source file already uses them or they are necessary.

### Python style

- Use syntax compatible with Python 3.10 or later and follow scikit-learn's code organization and naming conventions.
- Follow PEP 8 and the Black style used by scikit-learn. Keep line lengths, indentation, blank lines, naming, and import groups reasonable.
- Public modules, classes, functions, and methods must have docstrings.
- Write docstrings, templates, and code examples in English. Use the NumPy docstring style used by scikit-learn, with a concise summary and applicable `Parameters`, `Returns`, `Raises`, and `Examples` sections.
- Docstrings must accurately describe behavior, input types, return values, and exceptions. Do not copy boilerplate that does not match the implementation. Prefer reproducible minimal examples using the public API.
- Format string literals in Black style as used by scikit-learn, preferring double quotes. Use double-quoted f-strings for interpolation. Use single quotes only when a string contains double quotes or when they clearly improve readability.
- Group imports into standard library, third-party, and local package sections, keeping each group stable and clear.
- Keep each function focused and avoid passing data through implicit global state. Prefer structured APIs when working with pandas, NumPy, and scikit-learn objects.
- Preserve existing public interfaces unless the task explicitly requires a change. Update callers and tests alongside any interface change.

### Encoding and file formats

- Use UTF-8 for Python source files.
- Use LF (`\n`) line endings for text files; do not introduce unrelated CRLF changes.
- Preserve an existing Python encoding declaration such as `# -*- coding: utf-8 -*-`. Add one to new Python files when required by the project's conventions.
- Preserve or add author information in Python file headers, for example:

  ```python
  # -*- coding: utf-8 -*-

  __author__ = "Name <email@example.com>"
  ```

- Do not add copyright or license headers unless explicitly requested.
- When editing a file, preserve its existing encoding, line endings, and final newline. Avoid unrelated formatting changes.

## Testing requirements

- Put tests under `tests/`; do not add test files at the repository root.
- Use Python's standard-library `unittest`. Test classes must inherit from `unittest.TestCase`, and test methods must be named `test_*`.
- Add tests for new features and fixes. Prefer tests of public APIs and observable behavior over tests of implementation details.
- Use fixed, minimal, reproducible test inputs. Avoid reliance on the network, caches in a user's home directory, or undeclared external services.
- Before completion, run at least the tests for the affected module. Test discovery can be run with:

  ```shell
  python -m unittest discover -s tests -p 'test_*.py'
  ```

- When a test fails, determine whether the failure is caused by the change. Do not weaken assertions or delete tests just to make them pass.

## Version control

- Keep each commit focused on a complete, reviewable change, including implementation, necessary tests, and related documentation.
- Before committing, inspect `git diff`, `git diff --cached`, and `git status` to ensure user-unrequested files, generated artifacts, caches, and experimental results are not included.
- Use Conventional Commit messages in the format `<type>(<scope>): <imperative description>`, for example `feat(impute): add sklearn compatible imputation estimators`.
- Keep the commit subject short and accurate. Unless the change is very small, add a list of details after a blank line.
- Common types include `feat`, `fix`, `test`, `docs`, `refactor`, `perf`, `build`, and `chore`. Use the affected module or feature as the scope.
- Do not create commits unless explicitly instructed.
- Do not push unless explicitly instructed. Even after creating a commit, wait for explicit permission before pushing.
- Do not use destructive Git commands to overwrite or remove user changes. Confirm the scope before resolving conflicts or cleaning unrelated changes.
