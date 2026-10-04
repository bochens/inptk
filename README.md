# INP-toolkit (`inptk`)

INP-toolkit converts droplet-freezing observations into temperature-dependent
ice-nucleating particle (INP) concentrations. It provides a Python API, a command-line
interface, and a persistent process for applications such as Icescopy.

## Installation

### macOS

Download the Apple Silicon `.pkg` from [GitHub Releases](https://github.com/bochens/inptk/releases/latest).
The installer includes Python and the numerical libraries. The executable is:

```text
/Applications/INP-toolkit/inptk
```

Select this file in Icescopy's INP-toolkit executable preference. Restart the toolkit
process after installing an update. The current Mac installer is unsigned.

### Python

Requires Python 3.11 or newer:

```bash
git clone https://github.com/bochens/inptk.git
cd inptk
python -m pip install .
inptk --help
```

The [tutorial notebook](notebook/tutorial.ipynb) uses synthetic data to demonstrate
sample grouping, water blanks, concentration calculation, and plotting. To run it:

```bash
python -m pip install ".[plot]" jupyterlab
jupyter lab notebook/tutorial.ipynb
```

## Input

Native input accepts CSV files, pandas DataFrames, row records, or dictionaries of
column arrays. Counts identify a physical droplet set with `measurement_id`:

```csv
measurement_id,cycle_id,temperature_C,n_frozen,n_total
A,1,-5,0,32
A,1,-10,4,32
A,1,-15,16,32
B,1,-5,0,32
B,1,-10,1,32
B,1,-15,4,32
water,1,-5,0,32
water,1,-10,0,32
water,1,-15,1,32
```

Save these synthetic counts as `counts.csv`. Save the corresponding metadata as
`measurements.csv`:

```csv
measurement_id,sample_id,dilution,droplet_volume_uL
A,sample,1,50
B,sample,10,50
water,water,1,50
```

`sample_id` identifies the original sampled material; several droplet sets can
belong to it. `dilution` is the dilution factor, with 1 meaning undiluted.
Optional `run_id`, `time_s`, and `observation_id` preserve acquisition identity
and order. Repeated freezing cycles remain separate.

Air concentration also requires `sample_type="air"`, `air_volume_L`,
`suspension_volume_mL`, and `filter_fraction_used`. Soil concentration requires
`sample_type="soil"`, `dry_mass_g`, and `suspension_volume_mL`.

`read_icescopy()` accepts `freeze_count_timeseries.csv` and `.icescopy` projects,
including their exported metadata. Supply missing physical metadata through
`metadata=`. Sample grouping and water-blank assignment are explicit:

```python
import inptk

experiment = inptk.read_icescopy(
    "experiment.icescopy",
    sample_map={"A": "sample", "B": "sample", "water": "water"},
    water_blank_map={"A": ["water"], "B": ["water"]},
)
```

## Python workflow

```python
import inptk

experiment = inptk.read_counts(
    "counts.csv",
    metadata="measurements.csv",
    water_blank_map={"A": ["water"], "B": ["water"]},
)
curves = {"sample": {"inputs": ["A", "B"], "cycle": "1"}}
result = inptk.analyze_concentration(
    experiment,
    curves=curves,
    method="mle",
    temperature_step_C=0.5,
)

spectrum = result.curves["sample"].cumulative.to_dataframe()
result.save("analysis.inptk")
result.export_csv("concentrations.csv", curve_id="sample")
```

Results include the original counts, frozen fractions, concentration and uncertainty,
source observations, and excluded points. To request individual spectra, include
curves with a single input, for example `{"A": {"inputs": ["A"], "cycle": "1"}}`.

### Calculation methods

**Average** directly calculates each input's concentration, subtracts the assigned
blank concentration, and takes the arithmetic mean where input ranges overlap.
Binomial count uncertainty is propagated through those operations, including
shared blank uncertainty.

**MLE** uses maximum likelihood estimation: it jointly fits monotone sample and
blank curves to the complete selected freezing histories and calculates
concentration uncertainty from the likelihood.

Reported concentrations span the original sample's first through last freezing
events. Values and uncertainty outside the interval are `NaN`. Average contributors
must lie within each input's own interval and have positive frozen sample counts.
A zero concentration produced by blank correction remains valid inside the interval.

### Temperature ranges and grid

Average can suggest nonoverlapping ranges that preserve a nonnegative, nondecreasing
concentration as the sample cools:

```python
ranges = inptk.suggest_temperature_ranges(experiment, curves=curves, temperature_step_C=0.5)
result = inptk.analyze_concentration(
    experiment, curves=curves, method="average", temperature_step_C=0.5,
    temperature_ranges_C=ranges.temperature_ranges_C,
)
```

Manual limits use input names, for example
`{"A": {"min_C": -15, "max_C": -10}}`. They apply to combined curves.
MLE accepts manual ranges.

A grid selects counts before calculation. Omit `temperature_step_C` to retain native
observations. Optional `temperature_start_C` and `temperature_end_C` use the observed
limits when omitted. Temperature-selection methods are `latest`, `max`, and `window`;
`window` requires `temperature_window_C`. Use the same grid and settings for range
suggestions and analysis.

## Command-line workflow

```bash
inptk analyze counts.csv --metadata measurements.csv \
  --water-blank-map '{"A":["water"],"B":["water"]}' \
  --curves '{"sample":{"inputs":["A","B"],"cycle":"1"}}' \
  --method average --temperature-step-C 0.5 --out analysis.inptk
inptk table analysis.inptk --table cumulative
inptk export-csv analysis.inptk --out concentrations.csv
```

Use `--format icescopy` for an Icescopy CSV or project. `--output-basis sampled_air`
converts to INP per litre of sampled air when the required metadata are present.

Each step is also available separately: `fractions`, `estimate`, `convert`,
`finalize`, and `differentiate`. Use `suggest-ranges --summary` for Average limits.
`table` and `export-csv` access counts, frozen fractions, cumulative, excluded, and
differential tables. Run `inptk COMMAND --help` for each command's settings.

### Application clients

Start `inptk serve` once and exchange one JSON object per line through stdin/stdout.
Import raw counts and metadata into named `@references` and reuse them for analysis.
For example, after importing `@input`:

```json
{"id":1,"args":["analyze","@input","--format","saved","--method","average","--temperature-step-C","0.5","--out","@result"]}
{"id":2,"args":["table","@result","--table","cumulative","--no-history"]}
{"id":3,"release":["@result"]}
```

`capabilities` describes commands, defaults, the import schema, and number encoding.
JSON represents NaN as `{"$nonfinite":"nan"}`. Diagnostics use stderr.

## Development and help

Install development tools with `python -m pip install -e ".[dev]"` and run `pytest`.
See the [Mac build instructions](packaging/macos/README.md) and
[Windows build instructions](packaging/windows/README.md) for standalone installers.
Report problems or ask questions through [GitHub Issues](https://github.com/bochens/inptk/issues).

## Licence

Licensed under [AGPL-3.0](LICENSE). INP-toolkit builds on
[OLAF](https://doi.org/10.5281/zenodo.17509699).
