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
Start with `examples/standard_workflow.py` for a small synthetic example.
The [Icescopy-to-air-concentration notebook](notebook/icescopy_freeze_count_to_air_inp_demo.ipynb)
shows the current API step by step on the M1 dataset, including explicit sample
mapping, input checks, automatic stitching, MLE, air normalization, and an OLAF
comparison. It requires Jupyter, the `plot` extra, and the local source data at the
paths configured near the top. Its dataset-specific settings are documented there;
external result export is off by default.

## A standard analysis

```python
import inptk

experiment = inptk.read_counts("counts.csv", metadata="measurements.csv")
result = inptk.analyze_concentration(
    experiment,
    dilution_method="stitch",
    output_basis="sampled_air",
    decrease_policy="stop_at_decrease",
)

result.final.to_dataframe()                       # final combined concentration
result.per_dilution.to_dataframe()                # inspect individual dilutions
result.final_candidates.to_dataframe()           # inspect retained and excluded final points
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

## Work one step at a time

The full workflow calls these same public functions. You can stop after any step:

```python
fractions = inptk.frozen_fraction(experiment.counts)
individual = inptk.cumulative_spectrum(fractions, experiment=experiment)

combined = inptk.combine_dilutions(
    fractions,
    experiment=experiment,
    method=inptk.Stitch(min_unfrozen=3),
)
converted = inptk.convert_concentration(combined, experiment.samples, basis="sampled_air")
final = inptk.finalize_spectrum(converted, decrease_policy="stop_at_decrease")
```

Frozen fractions need only the labelled counts table. Concentration calculations
also need `experiment`, which supplies each measurement's dilution and droplet
volume. Combining receives the frozen-fraction table because methods such as
maximum likelihood estimation (MLE) use the observed counts, not just the
individual concentration curves. Each step returns one scientific table.
`table.history` records its processing steps; `table.warnings` reports unavailable
results or settings that could not apply.

`inptk.differential_spectrum(fractions, experiment=experiment)` is an optional
separate calculation. `inptk.subtract_blanks(combined, {"A": blank_spectrum})`
applies explicit blank correction before unit conversion. `finalize_spectrum`
then selects the final nondecreasing concentration rows without changing the
input table, concentrations, or error bounds. It returns one spectrum table.
The complete workflow also retains every candidate final row in
`result.final_candidates`, including the reason for each exclusion, and packages
all stages for saving with `result.save(...)`.

Differential concentration is the increase in cumulative concentration between
adjacent temperatures, divided by their actual temperature difference. Each
endpoint uses its own frozen fraction, so changing blank-corrected totals are
supported. Negative changes are retained with quality flag `2`; non-finite
results carry flag `1` (flags can combine). Each row stores both interval edges;
there is no invented interval before the first supplied state. This calculation
does not establish whether a change caused by lost droplets is unbiased.

## Choose how to combine dilutions

Each method has its own settings object. Pass it as `method=` to
`combine_dilutions`, or as `dilution_method=` to `analyze_concentration`.
The strings `"stitch"` and `"mle"` remain shortcuts for their default settings.
Unsupported settings raise an error instead of being ignored.

### Automatic stitching

```python
method = inptk.Stitch(min_unfrozen=5)
result = inptk.analyze_concentration(experiment, dilution_method=method)
```

- `min_unfrozen`: require at least this many unfrozen droplets at each eligible
  point. Default `3` retains the original exclusion of two or fewer unfrozen
  droplets. It must be a positive whole number. Missing and infinite
  concentrations are always excluded.

Automatic stitching orders curves from least to most diluted. It keeps the
current curve through its coldest eligible temperature and uses the next dilution
only at colder output temperatures. A missing point inside the current range does
not trigger an early switch or get filled from another curve.

Every retained temperature uses **one dilution's existing concentration and error
bounds unchanged**. There is no overlap setting, averaging, or joint refit during
stitching. `source_measurement_id` identifies the selected measurement; unavailable
points have an empty source. Final handling of decreases is a separate step.

A group with only one measurement uses its cumulative spectrum directly. There
is no dilution join in that case; customized stitching settings produce a warning
rather than being applied as a general concentration filter.

### Manual stitching

```python
method = inptk.ManualStitch(switch_temperatures_C=[-12, -18])
result = inptk.analyze_concentration(experiment, dilution_method=method)
```

For dilution factors `1`, `10`, and `100`, this selects:

| Temperature | Dilution used |
|---|---:|
| Warmer than -12 C | 1 |
| -18 C < temperature <= -12 C | 10 |
| Temperature <= -18 C | 100 |

Dilutions are ordered from least to most diluted. Switching temperatures must
be strictly ordered from warm to cold, with one fewer switch than dilutions.
At a switching temperature, the next dilution is used. A switch between grid
points takes effect at the first colder point; there is no interpolation.
An empty switch list is valid for a single dilution.

One configuration applies to all sample/run/cycle groups. They must have the same
observed dilution factors, with exactly one measurement at each factor. Missing
dilutions or duplicate measurements at a selected dilution raise an error; analyze
groups needing different dilution sets or switching temperatures separately.

Manual stitching applies no automatic unfrozen-droplet cutoff,
or averaging. Missing temperatures or non-finite estimates in the selected curve
remain missing, with warnings; another dilution is never substituted. Each output
row records `dilution_fold`, `source_measurement_id`, and `selection_status`.
`source_measurement_ids` lists all supplied measurements, not just the selected one.
Requested blank correction and unit conversion act on the combined spectrum.
The final `decrease_policy` can exclude rows from it; the combined curve remains
available unchanged for inspection.

### Joint count fitting (MLE)

MLE means maximum likelihood estimation: fit a concentration to the frozen and
total droplet counts from the different dilutions together.

```python
method = inptk.MLE(
    temperature_eligibility_C={"Sample_2": -15},
    mask_mode="drop_rows",
)
result = inptk.analyze_concentration(experiment, dilution_method=method)
```

This example keeps measurement `Sample_2` rows at -15 C and colder in the MLE fit.
All MLE dictionaries use the exact `measurement_id` as their key. For Icescopy
imports, this is the exported `sample_name`, such as `Sample_2`, rather than its
dilution factor or the parent sample name assigned by `sample_map` (such as
`CRG_M1`). Settings follow that measurement through all its cycles. Unknown
measurement names are rejected. Measurements omitted from temperature limits are
unrestricted; those omitted from weights have weight 1. Temperature limits require
an explicit choice of `mask_mode`:

- `"drop_rows"`: omit warmer rows but keep the original cumulative counts at
  retained temperatures.
- `"rebase_counts"`: also subtract the warm-side frozen baseline and remove those
  droplets from the total. This changes the scientific interpretation; select it
  only when deliberately excluding those warm freezing events.

Other retained MLE controls are `likelihood_weights` (positive relative
contributions to the fit), or `action_counts` with one of
`action_weight_lambda` or `action_weight_half_life` (an exponential weighting rule
based on action counts supplied by the caller). These two weighting approaches
cannot be combined. The package does not infer action counts from cycle numbers.
`confidence_drop` controls the decrease in log likelihood defining the uncertainty
interval; its default resolves to `z**2/2`. The effective value is saved.

Each temperature is fitted separately. The same droplets observed at different
temperatures are not treated as additional independent observations. Final
selection uses `decrease_policy`; it does not refit concentrations or narrow their
uncertainty bounds. There is no public `enforce_monotone` option.

MLE assumes independent physical droplet sets across its dilution inputs.
[Raw water-blank input](#raw-sample-and-water-blank-observations) fits the shared
background directly and restricts the weighting and masking controls accordingly.
Repeated cycles remain separate. With counts already corrected for a water blank, both the
retained OLAF count-based intervals and MLE intervals change because they use the
adjusted frozen and total counts. Neither separately propagates uncertainty in
the measured water blank or correlations introduced by a shared blank. Relative
weights also change the likelihood; weighted intervals should not be described
as ordinary droplet-count intervals.

### Select the final concentration curve

After dilution combination, any requested blank subtraction, and unit conversion,
`decrease_policy` selects concentration rows independently for each sample, run,
and cycle. It examines temperatures from warm to cold and compares each finite
concentration with the last retained value:

- `"stop_at_decrease"` (default): at the first strictly lower concentration,
  exclude that point and every colder point, even if the curve later recovers.
- `"skip_decreases"`: exclude a strictly lower point and continue checking colder
  points. Retain them if they equal or exceed the last retained value.

For concentrations `10, 12, 11, 13` in cooling order, the default retains `10, 12`;
`skip_decreases` retains `10, 12, 13`. Equal values are retained. Even a small
strict decrease triggers the rule: there is no hidden tolerance or window.
Nonfinite values are excluded and do not establish a comparison value. Negative
finite concentrations are not clipped; this selection rule alone does not
establish that they are scientifically usable.

Neither policy raises a concentration to make a plateau or changes error bounds.
`result.final` contains the retained rows. `result.final_candidates` contains all
rows after blank correction and unit conversion, with `used_in_final` and
`final_selection_status` to explain selection. Statuses are `kept`, `nonfinite`,
`decrease`, or `colder_than_decrease`. Earlier tables remain available.
This final selection is separate from selecting count observations by temperature
and from choosing which measurement rows enter the MLE fit.

```python
result = inptk.analyze_concentration(experiment, decrease_policy="skip_decreases")
```

### Command-line settings

The same settings are accepted as a JSON object or the path to a JSON file:

```bash
inptk analyze counts.csv --metadata measurements.csv --dilution-method stitch \
  --method-options '{"min_unfrozen":5}' --out automatic.inptk
inptk analyze counts.csv --metadata measurements.csv --dilution-method manual \
  --method-options '{"switch_temperatures_C":[-12,-18]}' --out manual.inptk
```

Resolved settings are saved in `result.settings["method_options"]` and table
history. Manual results also record the dilution order. Common temperature-grid,
output-unit, error-bar, and final-decrease settings remain workflow arguments.
Use `--decrease-policy stop_at_decrease` (default) or `--decrease-policy skip_decreases`.
Use `--sample A --cycle 01` to analyze only those exact labels; repeat either
option to select several. Sample selection uses the original sample name, while
MLE controls use measurement names. Unknown selections raise an error. Selection
and the retained source metadata are recorded in the saved experiment.

## Samples, measurements, cycles, dictionaries, and lists

An `Experiment` contains one `CountsTable`, sample and measurement dictionaries,
and an optional explicit water-blank assignment:

```python
experiment.samples["A"]                  # original sample and normalization inputs
experiment.measurements["A_neat"]        # physical droplet set and dilution
experiment.counts.select(sample_id="A")  # selected rows, still a CountsTable
experiment.water_blank_map               # raw sample measurement -> list of blank measurements
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

## Raw sample and water-blank observations

To include uncertainty in the water blank, provide **raw counts for both sample
and blank droplet sets** through native input. Each physical set has a separate
`measurement_id`, including each independent blank set. Counts may differ between
sets. Each set must have a known positive `droplet_volume_uL`; those volumes may
also differ. Do not recover raw counts by reversing an earlier correction.

Assign blanks explicitly; their names have no special meaning:

```python
experiment = inptk.read_counts(
    "raw_counts.csv",
    metadata="measurements.csv",
    water_blank_map={
        "A_neat": ["Water_1", "Water_2"],
        "A_diluted": ["Water_1", "Water_2"],
    },
)
result = inptk.analyze_concentration(experiment, dilution_method="mle")
```

The map always uses lists, even for one blank. It must assign every nonblank
measurement. Blank measurements have `dilution=1` and cannot also be sample
measurements. Assigned sets must belong to the same run. Dilutions of one original
sample in one run must use the same set of blanks; list order does not matter.
Different original samples may use different blank groups. Matching cycle and
temperature observations are required; cycles are never pooled.

For example, the corresponding measurement metadata can be:

```csv
measurement_id,sample_id,run_id,dilution,droplet_volume_uL
A_neat,A,1,1,50
A_diluted,A,1,10,50
Water_1,water1,1,1,20
Water_2,water2,1,1,50
```

`raw_counts.csv` contains each of these measurement names with its own
`cycle_id`, `temperature_C`, `n_total`, and `n_frozen`. Additional sample metadata
is required when converting suspension concentrations to air or soil units.

### The background model

The assigned sample and blank sets are assumed to share **one water-background
concentration per unit volume** at each temperature, from the same prepared-water
protocol. A larger droplet has a larger expected background contribution. This
is an explicit model assumption, not something the software can establish from
a blank name or droplet volume.

The fitted frozen probabilities are:

- Sample: `1 - exp[-V_sample * (C / dilution + B)]`.
- Blank set: `1 - exp[-V_blank * B]`.

Here `C` is concentration in the original sample suspension, `B` is the common
water-background concentration, both per mL, and each `V` is that set's droplet
volume in mL. The model describes randomly distributed ice-active contributions
per volume. It does not represent an additional background caused by well surface
area or differing preparation protocols.

When correction is enabled, individual dilution spectra fit the sample and its
assigned blanks together. Stitching selects one of those spectra at each
temperature, copying its concentration and uncertainty bounds unchanged. MLE fits
all eligible dilution counts and their common blank group together. In that joint fit, each independent blank set contributes once at a
temperature, regardless of how many dilutions reference it. Distinct volumes
remain in the probability model; unequal-volume blank counts are not collapsed
into one frozen fraction for calculation. Repeated cycles remain separate.

With raw water correction enabled, MLE supports temperature cutoffs with
`mask_mode="drop_rows"`. `temperature_method="latest"` or `"max"` is required;
`window_max_count` adds synthetic warm zero rows and cannot supply raw observations
for this fit. The window method remains available when correction is disabled or
no raw blank map is supplied.
It rejects `rebase_counts`, direct likelihood weights, and action-based weights:
those alter the raw-count interpretation used by this joint model. This route
also differs from the optional Python `blank_by_sample` operation, which subtracts
an already calculated sample/filter blank spectrum later in the workflow.

### Turn water correction on or off

Water correction is optional. With a map present, the default
`water_blank_correction=True` enables it. To analyze only the sample observations:

```python
result = inptk.analyze_concentration(experiment, water_blank_correction=False)
```

The same boolean is accepted by `cumulative_spectrum`, `combine_dilutions`, and
`differential_spectrum`. Disabling correction excludes mapped blank sets from
calculated sample tables and does not require matching blank temperatures. It
preserves every original raw observation and the map in `result.experiment`.
Saved settings record both the requested `water_blank_correction` value and
`water_blank_correction_applied`. Without a map, no water correction is applied.
This option does not reverse correction already present in imported counts.

Use `--no-water-blank-correction` for the same choice on the command line.

For CLI use, save the assignment as `water-blank-map.json`:

```json
{"A_neat": ["Water_1", "Water_2"], "A_diluted": ["Water_1", "Water_2"]}
```

```bash
inptk analyze raw_counts.csv --format native --metadata measurements.csv \
  --water-blank-map water-blank-map.json --sample A --cycle 1 \
  --dilution-method mle --out analysis.inptk
```

Sample/cycle selection retains the associated blank observations and metadata
without producing blank samples as analysis results. Saved experiments preserve
the assignment, so `--format saved` uses it directly and rejects an override.
The current `--format icescopy` path accepts already corrected exports and rejects
`--water-blank-map`; use raw native input for the joint model.

Numerical and regression checks cover the raw-blank calculations, optional
correction, saved results, and CLI behavior. These checks verify the implemented
model and software; they do not establish the frequency with which its uncertainty
intervals contain the true concentration across all experimental conditions.

## Icescopy integration

The following importer reads the existing count export, which may already contain
water correction. Its retained count-based uncertainty is an approximation; it
does not reconstruct raw sample/blank counts or infer blank uncertainty. For the
new Icescopy analysis dialog, pass raw session counts through the native contract
above and make the water-blank assignment in that dialog.

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
Only supplied, nonmissing fields replace the corresponding header values; unknown
measurement names are rejected. CLI overrides accept CSV or a JSON array of
records, for example `[{"sample_id":"Sample_0","well_volume_uL":50}]`.

Partially missing cycle labels are rejected rather than merged into a single
cycle. Zero-total observations are also rejected: exclude unusable observations
explicitly and retain the exclusion record, as demonstrated in the notebook.

An external application can invoke:

```bash
inptk analyze freeze_count_timeseries.csv --format icescopy \
  --sample-map sample_map.json --output-basis sampled_air --out analysis.inptk
```

The application supplies input files and reads the saved analysis or exported
CSV. No Icescopy GUI plugin is installed by this package. Connecting the
Icescopy interface is a separate integration task. The
[Icescopy handoff](ICESCOPY_HANDOFF.md) defines the executable contract, settings
placement, plots, and remaining integration work.

## Results and scientific methods

`AnalysisResult` keeps the original `experiment`, `settings`, `history`, and
`warnings`, alongside these tables:

| Attribute | Type | Meaning |
|---|---|---|
| `frozen_fraction` | `FrozenFractionTable` | Frozen counts and fractions at selected temperature thresholds |
| `per_dilution` | `CumulativeSpectrumTable` | Each dilution's concentration in the original suspension |
| `combined` | `CumulativeSpectrumTable` | Dilution-combined suspension concentration |
| `final_candidates` | `CumulativeSpectrumTable` or `None` | All rows after requested blank correction and unit conversion, with final-selection flags; present in new full-workflow results |
| `final` | `CumulativeSpectrumTable` | Rows retained under `decrease_policy`; concentrations and error bounds are unchanged |
| `differential` | `DifferentialSpectrumTable` or `None` | Optional per-measurement activity per degree, with interval limits |

`dilution_method="stitch"` selects one dilution per output temperature using the
coldest eligible handoff described above.
`"mle"` fits the count observations jointly using the retained binomial-Poisson
method. Differential spectra are optional (`differential=True`); they are not an
intermediate step required for cumulative concentration.

Temperature selection runs separately for each measurement and cycle, using
observations through the first coldest temperature. Choose one rule with
`temperature_method`:

| Method | Which observation supplies the frozen and total counts? |
| --- | --- |
| `max` | Highest frozen **fraction** among observations at the target temperature or warmer, allowing the tolerance. This carries earlier peaks forward. |
| `latest` | Latest qualifying observation in time; coldest qualifying observation when timestamps are absent. Decreases remain visible. |
| `window_max_count` | Highest **frozen count** strictly inside the temperature tolerance window. If the window is empty, use the highest frozen count warmer than its upper edge. |

The selected frozen count and total count stay together. Ties for either maximum
use the latest qualifying observation. `step_C` sets the output temperature
spacing; `temperature_tolerance_C` sets the allowance around each target.
Defaults are `latest`, 0.5 degree C spacing, and zero tolerance: use the last
observation at or warmer than the target. Explicit `max` defaults to 0.05 degree C
tolerance; `window_max_count` defaults to 0.01 degree C tolerance. An explicit
tolerance overrides the method default in both Python and the command line.

```python
fractions = inptk.frozen_fraction(
    experiment,
    temperature_method="window_max_count",
    step_C=0.5,
    temperature_tolerance_C=0.01,
)
```

`window_max_count` adapts the count-selection rule in original OLAF's
[`SpacedTempCSV.create_temp_csv`](https://github.com/SiGran/OLAF/blob/970896f46e2aa50c1ce57ec38dee3ea3f305a615/olaf/processing/spaced_temp_csv.py).
It retains the first freezing observation rounded to 0.1 degree C, inserts four
warmer zero rows, and then advances by `step_C`, stopping before the coldest
observed temperature. OLAF uses the least-diluted measurement's first freezing
observation to start a shared table and applies water-background correction
later. INP-toolkit processes each measurement and cycle separately using the
counts supplied by the caller, which may already be corrected. This option
therefore does not reproduce the entire original OLAF workflow. The former
option name `olaf` has been replaced by `window_max_count`.

Concentration columns are `concentration`, `unit`, and `basis`. `lower_error`
and `upper_error` are error-bar widths: interval endpoints are concentration
minus/plus these widths. Their calculation method is recorded. They do not
include variability across repeated cycles. Non-finite estimates remain in
candidate and intermediate tables with their quality flags and are excluded from
`final`; inspect flags before plotting or further analysis.

Optional `blank_by_sample={"A": blank_spectrum}` maps each target sample to a
single blank sample's cumulative suspension spectra. Matching is exact by run,
cycle, and temperature. Missing coverage is an error; no extrapolation is done
implicitly. The retained root-sum-of-squares error propagation assumes independent
sample and blank errors. Corrected values may be negative and are not clipped.
The final `decrease_policy` selects rows after correction and unit conversion,
while retaining every corrected candidate for inspection.

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
  methods.py        automatic/manual stitching and MLE settings
  processing.py     separately callable count-to-fraction and spectrum steps
  workflows.py      dilution combination, correction, units, and full workflow
  io.py             versioned saving/loading of complete analyses
  cli.py            command-line arguments calling the same workflow
  __main__.py       python -m inptk
  _engine/          retained numerical methods and their internal working tables
```

The private engine contains the numerical implementations behind the public
workflow and scientific tables. New applications should use
`inptk`'s public imports; they should not import `_engine`. Individual-droplet
freezing-event analysis and GUI development are not yet implemented.

Saved analyses use format version 1 in `analysis.json`. They include the writing
package version (`toolkit_version`), metadata, identifiers, tables, settings, and
history; loading does not execute code. Non-finite numbers are explicitly
encoded, rather than silently changing
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
