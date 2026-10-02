# INP-toolkit (`inptk`)

INP-toolkit processes droplet-freezing observations into temperature-dependent
ice-nucleating particle (INP) concentrations. It accepts ordinary CSV/Python
tables and Icescopy exports. The same workflow is available from Python and the
command line, without importing or installing Icescopy.

## Install

```bash
python -m pip install -e ".[dev]"
inptk --help
```

The distribution, Python import, and command are named `inptk`. There is no
`ufolaf` compatibility package. Historical workflows remain in Git history.
The existing `notebook/icescopy_freeze_count_to_air_inp_demo.ipynb` documents the
old UFOLAF interface and is not a runnable INP-toolkit example. Start with
`examples/standard_workflow.py` for the current interface.

## A standard analysis

```python
import inptk

experiment = inptk.read_counts("counts.csv", metadata="measurements.csv")
result = inptk.analyze_concentration(
    experiment,
    dilution_method="stitch",
    output_basis="sampled_air",
)

result.final.to_dataframe()                       # final combined concentration
result.per_dilution.to_dataframe()                # inspect individual dilutions
result.frozen_fraction.to_dataframe()             # inspect the count reduction
result.final.select(sample_id="A", cycle_id="2") # returns another spectrum table
result.save("analysis.inptk")                     # includes original observations
result.export_csv("final_concentrations.csv")     # includes sample/run/cycle IDs
restored = inptk.load("analysis.inptk")
```

Existing output paths are not overwritten. Python and command-line analysis use
the same defaults and calculations:

```bash
inptk analyze counts.csv --metadata measurements.csv \
  --dilution-method stitch --output-basis sampled_air --out analysis.inptk
inptk export-csv analysis.inptk --out final_concentrations.csv
```

## Samples, measurements, cycles, dictionaries, and lists

An `Experiment` contains one `CountsTable`, plus two dictionaries:

```python
experiment.samples["A"]                  # original sample and normalization inputs
experiment.measurements["A_neat"]        # physical droplet set and dilution
experiment.counts.select(sample_id="A")  # selected rows, still a CountsTable
```

- `sample_id` identifies the original sampled material.
- `measurement_id` identifies one physical droplet set at a specified dilution.
  It stays the same across repeated freezing cycles. It is unique within an Experiment.
- `run_id` identifies a measurement session. Different runs are never implicitly joined.
- `cycle_id` identifies a freezing cycle within a run. Labels are text, including
  leading zeros. Omitting the cycle column in native input declares a single
  cycle labelled `1`; the reader does not detect cycles from temperature changes.

Sample and cycle observations stay in labelled table rows, not nested dictionaries.
Every final curve belongs to one sample, run, and cycle. Different dilutions are
combined within that group. Repeated cycles are never summed or averaged, and
there is no cycle-policy option. Missing dilution measurements in a cycle are
reported in `result.warnings`; available measurements are processed separately.

Lists are used for requested selections, warnings, and ordered processing history:

```python
selected = result.final.select(sample_id=["A", "B"], cycle_id=["1", "2"])
```

Selecting one or many labels returns the same table type. `to_dataframe()` returns
a copy, so editing it does not change the stored result.

## Native input

`counts.csv`:

```csv
measurement_id,cycle_id,time_s,temperature_C,n_total,n_frozen
A_neat,1,0,-5,32,0
A_neat,1,60,-6,32,4
A_diluted,1,0,-5,32,0
A_diluted,1,60,-6,32,1
```

`measurements.csv`, with one row per physical droplet set:

```csv
measurement_id,sample_id,run_id,dilution,droplet_volume_uL,sample_type,air_volume_L,suspension_volume_mL,filter_fraction_used
A_neat,A,1,1,50,air,100,5,1
A_diluted,A,1,10,50,air,100,5,1
```

`dilution=10` denotes a tenfold dilution; `1` is undiluted. Sample-level inputs
must agree across its dilution measurements. Soil normalization uses
`sample_type=soil`, `suspension_volume_mL`, and `dry_mass_g`. Suspension output
needs only droplet volume and dilution. `time_s` is optional, but should be
provided for observations containing both cooling and subsequent warming.

Both reader arguments also accept pandas DataFrames. Direct construction with
`CountsTable`, `SampleMetadata`, `MeasurementMetadata`, and `Experiment` is
available for Python callers.

## Icescopy integration

```python
experiment = inptk.read_icescopy(
    "freeze_count_timeseries.csv",
    sample_map={"A_1": "A", "A_10": "A"},
    run_id="run-01",
)
result = inptk.analyze_concentration(experiment, output_basis="sampled_air")
```

`sample_map` explicitly maps Icescopy measurement labels to original samples.
Without it, each label remains a separate sample. The importer never guesses
relationships by stripping numbers from names. Icescopy's header metadata must
provide the droplet volume and dilution, or the reader must receive metadata
overrides (`metadata=` in Python or `--metadata` on the command line). Overrides
use Icescopy measurement labels as `sample_id` and `well_volume_uL` for droplet volume.

An external application can invoke:

```bash
inptk analyze freeze_count_timeseries.csv --format icescopy \
  --sample-map sample_map.json --output-basis sampled_air --out analysis.inptk
```

The application supplies input files and reads the saved analysis or exported
CSV. No Icescopy GUI plugin is installed by this package. Connecting the
Icescopy interface is a separate integration task.

## Results and scientific methods

`AnalysisResult` keeps the original `experiment`, `settings`, `history`, and
`warnings`, alongside these tables:

| Attribute | Type | Meaning |
|---|---|---|
| `frozen_fraction` | `FrozenFractionTable` | Frozen counts and fractions at selected temperature thresholds |
| `per_dilution` | `CumulativeSpectrumTable` | Each dilution's concentration in the original suspension |
| `combined` | `CumulativeSpectrumTable` | Dilution-combined suspension concentration |
| `final` | `CumulativeSpectrumTable` | Requested blank correction and concentration units applied |
| `differential` | `DifferentialSpectrumTable` or `None` | Optional per-measurement activity per degree, with interval limits |

`dilution_method="stitch"` retains OLAF's dilution-transition calculation.
`"mle"` fits the count observations jointly using the retained binomial-Poisson
method. Differential spectra are optional (`differential=True`); they are not an
intermediate step required for cumulative concentration.

Temperature reduction defaults to a 0.5 degree C grid, the `max` method, cooling
observations only, and a 0.05 degree C tolerance. The explicit `olaf` reduction
method defaults to its 0.01 degree C tolerance. A supplied tolerance overrides
that choice for both Python and command-line use.

Concentration columns are `concentration`, `unit`, and `basis`. `lower_error`
and `upper_error` are error-bar widths: interval endpoints are concentration
minus/plus these widths. Their calculation method is recorded. They do not
include variability across repeated cycles. Non-finite estimates retain their
quality flags; values must be inspected before plotting or further analysis.

Optional `blank_by_sample={"A": blank_spectrum}` maps each target sample to a
single blank sample's cumulative suspension spectra. Matching is exact by run,
cycle, and temperature. Missing coverage is an error; no extrapolation is done
implicitly. The retained root-sum-of-squares error propagation assumes independent
sample and blank errors. Corrected values may be negative; this workflow does
not silently clamp them or apply additional quality-control adjustments.

`output_basis` is `suspension`, `sampled_air`, or `dry_soil`. Changing this basis
returns the same cumulative table type. Converting an already normalized result
again is rejected.

## Python file structure

```text
src/inptk/
  __init__.py       public imports
  tables.py         the four scientific table types and their checks
  experiment.py     sample/measurement information, Experiment, AnalysisResult
  readers.py        native CSV/Python and Icescopy import
  workflows.py      complete concentration analysis and unit conversion
  io.py             versioned saving/loading of complete analyses
  cli.py            command-line arguments calling the same workflow
  __main__.py       python -m inptk
  _engine/          retained numerical methods and their internal working tables
```

The private engine preserves the existing numerical implementations while the
public workflow and data structures are replaced. New applications should use
`inptk`'s public imports; they should not import `_engine`. Individual-droplet
freezing-event analysis and GUI development are not yet implemented.

Saved analyses use format version 1 in `analysis.json`. They include metadata,
identifiers, tables, settings, and history; loading does not execute code. IEEE
non-finite numbers are explicitly encoded, rather than silently changing
infinite concentrations into missing values.

## Origin and licence

INP-toolkit is the successor to UFOLAF, an unofficial fork of OLAF (OpenSource
Library for Automating Freezing data acquisition from Ice Nucleation
Spectrometer). It is independently maintained and is not affiliated with or
endorsed by the original OLAF authors. Original documentation is available at
<https://sigran.github.io/OLAF/>. Please also cite the original OLAF release when
using this work in research: <https://doi.org/10.5281/zenodo.17509699>.

Original copyright and licence notices are retained. The project remains
licensed under AGPL-3.0; renaming or describing an integration as a plugin does
not change that licence.
