# Command-line interface

SinComp provides two command-line interfaces: the `sincomp` command is
the human-friendly interface, while the `sincomp.cli.*` modules provide
a stable, agent-oriented interface.

Install SinComp to use either interface:

```shell
python -m pip install sincomp
sincomp --help
```

The human-friendly command groups common dialect-data workflows. Its
commands are dispatched to the corresponding `sincomp.cli.*` module:

| Human-friendly command | Agent-oriented module | Purpose |
| --- | --- |
| `sincomp dataset` | `sincomp.cli.dataset` | List built-in datasets, inspect their dialects, or query pronunciation records. |
| `sincomp similarity` | `sincomp.cli.similarity` | Calculate pairwise dialect similarities. |
| `sincomp compare` | `sincomp.cli.compare` | Calculate how well dialect data comply with pronunciation rules. |
| `sincomp align` | `sincomp.cli.align` | Align character IDs across datasets, or evaluate alignment accuracy. |
| `sincomp classifier` | `sincomp.cli.classifier` | Train, evaluate, or use a dialect classifier. |
| `sincomp reconstruct` | `sincomp.cli.reconstruct` | Reconstruct proto-language phonemes, infer descendant pronunciations, or evaluate reconstruction. |

Agents can import a module and call its `main(argv)` function with the
arguments as a sequence of strings. For example:

```python
from sincomp.cli import dataset

exit_code = dataset.main(["list", "--format", "json"])
```

Each module prints its results and returns an exit code. The module
interface accepts the same command-specific arguments shown by its
help; unlike the human-friendly dispatcher, it can also expose explicit
output-format options.

Most analyses read a registered dataset, a local CSV file, or a dataset
directory; they write result files to the current directory unless an
output path is supplied.

For example, list datasets or compute one similarity measure:

```shell
sincomp dataset list
sincomp similarity --method chi2 MCPDict
```

Each command provides its own help. Use `--help` to see available
actions, options, defaults, and expected inputs:

```shell
sincomp dataset --help
sincomp classifier train --help
sincomp reconstruct evaluate --help
python -m sincomp.cli.dataset --help
```

Some workflows have nested actions, such as `align evaluate`,
`classifier train`, and `reconstruct inference`. Consult the help for
the selected action for required files, output formats, and configuration
options. Dataset availability and optional dependencies are described
in [Datasets](datasets.md) and [Installation](installation.md).
