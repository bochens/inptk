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
mapping, input checks, temperature ranges, concentration estimation, and air
normalization. It requires Jupyter, the `plot` extra, and the local source data at the
paths configured near the top. Its dataset-specific settings are documented there;
external result export is off by default.

## A standard analysis

```python
import inptk

experiment = inptk.read_counts("counts.csv", metadata="measurements.csv")
result = inptk.analyze_concentration(
    experiment,
    method="mle",
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
  --output-basis sampled_air --out analysis.inptk
inptk export-csv analysis.inptk --out final_concentrations.csv
```

For the CSU CSV layout, the repository helper formats an already saved final
sampled-air spectrum:

```bash
python scripts/csu_inp_processing.py analysis.inptk --out csu.csv \
  --sample A --run 1 --cycle 1 --allow-missing-header
```

Selection flags can be omitted when only one sample/run/cycle remains. Use
`--header KEY=VALUE` for CSU header fields; `--allow-missing-header` leaves
unavailable descriptive fields blank. Recorded normalization metadata cannot be
changed. The exporter preserves the saved final points and errors without
recalculating concentrations. Its columns are `degC`, `dilution`, `INPS_L`,
`lower_CI`, and `upper_CI`; the two CI columns contain **error widths**, not
interval endpoints. Points combining several dilutions leave `dilution` blank.
Outputs must be new paths outside the saved analysis folder.

## Work one step at a time

The full workflow calls these same public functions. You can stop after any step:

```python
ranges = {"A_neat": {"min_C": -15}}
fractions = inptk.frozen_fraction(experiment)
individual = inptk.cumulative_spectrum(
    fractions, experiment=experiment, temperature_ranges_C=ranges
)

combined = inptk.combine_dilutions(
    fractions,
    experiment=experiment,
    method="mle",
    temperature_ranges_C=ranges,
)
converted = inptk.convert_concentration(combined, experiment.samples, basis="sampled_air")
final = inptk.finalize_spectrum(converted, decrease_policy="stop_at_decrease")
```

Frozen fractions need only the labelled counts table. Concentration calculations
also need `experiment`, which supplies each measurement's dilution and droplet
volume. Combining receives the frozen-fraction table because it needs the frozen
and total droplet counts stored there. Those counts supply both concentration
estimates and their uncertainty. Each step returns one scientific table.
`table.history` records its processing steps; `table.warnings` reports unavailable
results or settings that could not apply.

For optional differential output, call `differential_spectrum` with the same
fractions, experiment, and temperature ranges.
`inptk.subtract_blanks(combined, {"A": blank_spectrum})` subtracts a calculated
sample/filter blank spectrum before unit conversion. `finalize_spectrum`
then selects the final nondecreasing concentration rows without changing the
input table, concentrations, or error bounds. It returns one spectrum table.
The complete workflow also retains every candidate final row in
`result.final_candidates`, including the reason for each exclusion, and packages
all stages for saving with `result.save(...)`.

Differential concentration is the increase in cumulative concentration between
adjacent temperatures, divided by their actual temperature difference. Each
endpoint uses its own concentration estimate, so changing blank-corrected totals
are supported. Missing or excluded endpoints produce a missing interval; the
calculation does not bridge gaps. Negative changes carry quality flag `2`;
non-finite results carry flag `1` (flags can combine). Each row stores both interval edges;
there is no invented interval before the first supplied state. This calculation
does not establish whether a change caused by lost droplets is unbiased.

## Combine dilutions within explicit temperature ranges

Use `method="mle"` (default) or `method="average"` with the same temperature ranges:

- **MLE**, maximum likelihood estimation, finds the concentration most consistent
  with the eligible measurements' frozen and total droplet counts together.
- **Average** calculates each eligible measurement's concentration in the original
  suspension, then takes their equal-weight arithmetic mean.

At each temperature, only measurements of the same original sample, run, and
cycle are combined. With one eligible measurement, both choices return that
measurement's concentration and uncertainty. With none, neither invents a value.
There is no separate stitching operation.

```python
result = inptk.analyze_concentration(
    experiment,
    method="mle",  # use "average" to average per-measurement concentrations
    temperature_ranges_C={
        "Sample_0": {"min_C": -15},
        "Sample_1": {"min_C": -22, "max_C": -12},
    },
)
```

Each key is an exact `measurement_id`. For Icescopy imports, this is the exported
`sample_name`, such as `Sample_0`, rather than a dilution factor or the parent
sample name assigned by `sample_map`.

- `min_C` is the cold limit; `max_C` is the warm limit. Both endpoints are included.
  For example, `{"min_C": -22, "max_C": -12}` retains −22°C through −12°C.
- Omit either bound, or set it to `None` in Python / `null` in JSON, for no limit
  on that side. Omitted measurements use their full available temperature range.
- Overlapping ranges use every eligible measurement, through the chosen method.
  Disjoint ranges can restrict each output temperature to one measurement. A
  temperature with no eligible measurement has no concentration estimate.
- Limits apply to the selected output temperatures and follow a measurement
  through its cycles. They do not delete source observations or subtract frozen
  events from the remaining cumulative counts. Unknown IDs and reversed limits
  raise errors.

No hidden minimum-unfrozen-droplet cutoff or automatic range selection is applied.
Future tools may suggest ranges, but should return this same explicit mapping for
review. To use different ranges for different cycles, analyze those cycles in
separate jobs.

The same `temperature_ranges_C` keyword is accepted by `cumulative_spectrum`,
`combine_dilutions`, and `differential_spectrum`. Individual concentration rows
outside a range retain their count columns but have missing concentration/error
values and `selection_status="outside_temperature_range"`. The frozen-fraction
table remains complete. Missing blank coverage outside a sample's selected range
does not block analysis; eligible temperatures still require blank coverage.

The count model assumes independent physical droplet sets across measurements.
Each temperature is estimated separately: seeing the same droplets at another
temperature or in another freezing cycle does not create extra independent
observations. Repeated cycles remain separate throughout the workflow.

MLE uncertainty uses a profile-likelihood interval: keep concentrations that remain
sufficiently consistent with the observed counts after allowing any fitted water
background to vary. The log-likelihood threshold is `z**2 / 2`; the default
`z=1.96` gives nominal 95% bounds. The rule is the same for one or many eligible
measurements. The bounds describe count uncertainty under the model, not
cycle-to-cycle variability or uncertainty in supplied volumes and dilutions.

Average uses conservative bounds that allow shared blank uncertainty; MLE uses
counts jointly. For `average`, the uncertainty label is **Bonferroni-adjusted
marginal profile bounds**. Each measurement is estimated with its assigned raw
blank observations, then its interval is widened before the endpoints are averaged:

1. Convert `z` to the nominal probability of falling outside the interval, called
   `alpha`. Divide it by the number of eligible measurements, `m`.
2. Calculate each measurement's profile bounds using this smaller `alpha / m`.
3. Average all lower endpoints, and separately average all upper endpoints.

The [Bonferroni adjustment](https://www.itl.nist.gov/div898/handbook/prc/section4/prc463.htm)
does not require independent intervals, so a shared blank is not mistaken for
independent background information. The individual profile intervals remain
approximate: this is not a promise of exact 95% coverage. Bounds may stay wide
or widen when more measurements contribute; averaging does not guarantee smaller
error bars. With one measurement, its ordinary MLE bounds are used unchanged.
An infinite individual estimate makes the mean infinite; an unavailable individual estimate
makes the mean unavailable. Neither is silently omitted from the average.

Both methods include measured background uncertainty when raw blank
correction is enabled. Raw sample and blank counts allow that background to be
estimated, as described [below](#raw-sample-and-water-blank-observations).
Counts that were corrected before import cannot recover the original blank's
uncertainty. Their intervals are conditional on the supplied adjusted counts.

### Select the final concentration curve

After dilution combination, any requested blank subtraction, and unit conversion,
`decrease_policy` selects concentration rows independently for each sample, run,
and cycle. It examines temperatures from warm to cold and compares each finite
concentration with the last retained value:

- `"stop_at_decrease"` (default): at the first lower concentration beyond the
  numerical-equality margin below, exclude that point and every colder point,
  even if the curve later recovers.
- `"skip_decreases"`: exclude a lower point beyond that margin and continue
  checking colder points. Retain them if they equal or exceed the last retained
  value within numerical precision.

For concentrations `10, 12, 11, 13` in cooling order, the default retains `10, 12`;
`skip_decreases` retains `10, 12, 13`. Equality allows a fixed relative numerical
margin of `1e-9`, with zero absolute margin, to avoid treating solver roundoff as
a decrease. Outside this tiny margin, a decrease triggers the rule. This is not
a user setting or a smoothing window; values and error bounds stay unchanged.
The numerical margin is recorded in selection history.
Nonfinite values are excluded and do not establish a comparison value. Negative
finite concentrations are not clipped; this selection rule alone does not
establish that they are scientifically usable.

Neither policy raises a concentration to make a plateau or changes error bounds.
`result.final` contains the retained rows. `result.final_candidates` contains all
rows after blank correction and unit conversion, with `used_in_final` and
`final_selection_status` to explain selection. Statuses are `kept`, `nonfinite`,
`decrease`, or `colder_than_decrease`. Earlier tables remain available.
This final selection is separate from selecting count observations by temperature
and from choosing which measurement rows enter the combination.

```python
result = inptk.analyze_concentration(experiment, decrease_policy="skip_decreases")
```

### Command-line settings

Supply the temperature ranges as a JSON object or the path to a JSON file:

```bash
inptk analyze counts.csv --metadata measurements.csv \
  --temperature-ranges '{"Sample_0":{"min_C":-15},"Sample_1":{"max_C":-12}}' \
  --out analysis.inptk
```

The equivalent Python keyword is `temperature_ranges_C`. Resolved ranges are
saved in `result.settings["temperature_ranges_C"]` and table history;
`result.settings["estimation_method"]` records `"mle"` or `"average"`. Use
`--method average` to select the mean; omitting `--method` uses MLE.
Temperature-grid, output-unit, uncertainty, and final-decrease settings remain
workflow arguments. Use `--decrease-policy stop_at_decrease` (default) or
`--decrease-policy skip_decreases`.

Use `--sample A --cycle 01` to analyze only those exact labels; repeat either
option to select several. Sample selection uses the original sample name, while
temperature ranges use measurement names. Unknown selections raise an error.
Selection and retained source metadata are recorded in the saved experiment.

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
result = inptk.analyze_concentration(experiment)
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

The assigned blanks measure the combined background from water, substrate,
PCR well walls, and other assay sources. INP-toolkit represents all of these as
**one shared background per unit liquid volume** at each temperature. It assumes
the entire background contribution scales with liquid volume: a larger droplet
has a proportionally larger expected contribution.

This is a simplifying physical assumption. The model does not separate the
individual background sources. Assigned blanks should represent the sample's
assay preparation. Each sample and blank uses its own actual droplet volume;
their volumes do not have to be equal. There is one model and no model selector.
Saved settings record this fixed assumption as `water_blank_model="volume_scaled"`.

The fitted frozen probabilities are:

- Sample: `1 - exp[-V_sample * (C / dilution + B)]`.
- Blank set: `1 - exp[-V_blank * B]`.

Here `C` is concentration in the original sample suspension, `B` is the combined
background concentration, both per mL of liquid, and each `V` is that set's actual
droplet volume in mL. The model describes randomly distributed ice-active
contributions per liquid volume. Water, substrate, and wall contributions all
enter through `B`; there is no separate surface-area term.

MLE fits this common assay background explicitly. It cannot identify or remove
contamination unique to one dilution and absent from its assigned blanks.
Measurement-specific temperature ranges remain an explicit choice by the user.

When correction is enabled, individual dilution spectra estimate sample
concentration from that measurement and its assigned blanks. The combined
spectrum uses all measurements allowed by `temperature_ranges_C`. MLE fits their
counts with the common blank group; average uses their individually estimated
concentrations. In an MLE fit, each independent blank set contributes once at a temperature,
regardless of how many dilutions reference it. Distinct droplet volumes remain
in the probability model; unequal-volume blank counts are not collapsed into
one frozen fraction. Repeated cycles remain separate.

A blank with 32 wells is not subtracted as 32 droplets from a sample with four
wells. Each count is interpreted with its own observed total and droplet volume.
At the same frozen fraction and volume, more blank wells imply the same background
level, with greater statistical precision. The joint fit uses that stronger
information; the number of wells is not a multiplier for the correction.

All concentration calculations require `temperature_method="latest"` or `"max"`.
`window_max_count` inserts synthetic warm zero rows and is available only for
frozen-fraction previews. Temperature ranges select whole count observations;
there are no count-rebasing or arbitrary contribution-weight settings. This model
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
  --out analysis.inptk
```

Sample/cycle selection retains the associated blank observations and metadata
without producing blank samples as analysis results. Saved experiments preserve
the assignment, so `--format saved` uses it directly and rejects an override.
The current `--format icescopy` adapter does not carry raw blank context and
rejects `--water-blank-map`; use raw native input for the joint model.

Numerical and regression checks cover the raw-blank calculations, optional
correction, saved results, and CLI behavior. These checks verify the implemented
model and software; they do not establish the frequency with which its uncertainty
intervals contain the true concentration across all experimental conditions.

## Icescopy integration

The following importer reads the existing count export, which may already contain
water correction. Its uncertainty is conditional on the supplied counts; it
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

`combined` combines the eligible measurements at each temperature with the
requested `method`.
It records `contributing_measurement_ids`, `contributor_count`, and
`selection_status` (`single`, `combined`, or `no_eligible_measurements`).
`available_measurement_ids` lists measurements present before applying ranges;
`source_measurement_ids` lists the whole group's candidates. ID lists are JSON
strings. `source_measurement_id` is populated only when one measurement contributes.
These fields explain the calculation without treating missing data as a zero.
Differential spectra are optional (`differential=True`); they are not an
intermediate step required for cumulative concentration.

Temperature selection runs separately for each measurement and cycle, using
observations through the first coldest temperature. `latest` and `max` can supply
concentration calculations; `window_max_count` is restricted to fraction previews:

| Method | Which observation supplies the frozen and total counts? |
| --- | --- |
| `max` | Highest frozen **fraction** among observations at the target temperature or warmer, allowing the tolerance. This carries earlier peaks forward. |
| `latest` | Latest qualifying observation in time; coldest qualifying observation when timestamps are absent. Decreases remain visible. |
| `window_max_count` (fraction preview only) | Highest **frozen count** strictly inside the temperature tolerance window. If the window is empty, use the highest frozen count warmer than its upper edge. |

The selected frozen count and total count stay together. Ties for either maximum
use the latest qualifying observation. `step_C` sets the output temperature
spacing; `temperature_tolerance_C` sets the allowance around each target.
Defaults are `latest`, 0.5 degree C spacing, and zero tolerance: use the last
observation at or warmer than the target. Explicit `max` defaults to 0.05 degree C
tolerance; `window_max_count` defaults to 0.01 degree C tolerance. An explicit
tolerance overrides the method default. The concentration CLI offers `latest`
and `max`.

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
  methods.py        combination-choice and temperature-range validation
  processing.py     separately callable count-to-fraction and spectrum steps
  workflows.py      dilution combination, correction, units, and full workflow
  io.py             versioned saving/loading of complete analyses
  cli.py            command-line arguments calling the same workflow
  __main__.py       python -m inptk
  water_blank.py    raw sample/blank model preparation and shared-background fits
  _engine/          numerical models and their internal working tables
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
