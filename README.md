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

## Preview observations before calculating concentrations

Counts and frozen fractions do not need dilution, droplet volume or air volume:

```python
import inptk

counts = inptk.read_observations("counts.csv")
fractions = inptk.frozen_fraction(counts)
```

For an Icescopy CSV use `format="icescopy"`. Optional `metadata=` can be incomplete
at this stage. Available metadata are retained in the count table's history.
Missing parent-sample assignments keep measurements separate; they are not
inferred from names. The concentration readers below still require complete
physical metadata and reject invalid inputs.

## A standard analysis

```python
import inptk

experiment = inptk.read_counts("counts.csv", metadata="measurements.csv")
result = inptk.analyze_concentration(
    experiment,
    curves={"A": {"inputs": ["A_neat", "A_diluted"], "cycle": "1"}},
    method="mle",
    output_basis="sampled_air",
    decrease_policy="stop_at_decrease",
)

curve = result.curves["A"]
curve.cumulative.to_dataframe()                  # retained concentration and uncertainty
curve.sources                                    # physical inputs used for this curve
curve.excluded.to_dataframe()                     # excluded points and their reasons
result.counts.to_dataframe()                      # every original count observation
result.frozen_fraction.to_dataframe()             # observed fractions, including blanks
result.to_dataframe()                             # cumulative rows from all named curves
result.save("analysis.inptk")                     # original data, choices and named curves
result.export_csv("A.csv", curve_id="A")          # export one curve
restored = inptk.load("analysis.inptk")
```

You choose output names. A single input produces an individual curve; several
independent inputs produce a combined curve, with equal or different dilution
factors. To keep both, request them in the same dictionary:

```python
curves = {
    "A": {"inputs": ["A_neat", "A_diluted"], "cycle": "1"},
    "A_neat": {"inputs": ["A_neat"], "cycle": "1"},
    "A_diluted": {"inputs": ["A_diluted"], "cycle": "1"},
}
result = inptk.analyze_concentration(experiment, curves=curves)
```

The output name is a label, not another sample or another independent observation.
Cycle can be omitted only when every named input has exactly one observed cycle.
Otherwise it must be explicit. Omitting `curves` generates names and keeps the
default separation by original sample, run and cycle. Only requested curves are
calculated; no redundant combined result is added for a single input.

Existing output paths are not overwritten. Python and command-line analysis use
the same defaults and calculations:

```bash
inptk analyze counts.csv --metadata measurements.csv \
  --curves '{"A":{"inputs":["A_neat","A_diluted"],"cycle":"1"}}' \
  --output-basis sampled_air --out analysis.inptk
inptk export-csv analysis.inptk --out final_concentrations.csv
```

For the CSU CSV layout, the repository helper formats an already saved final
sampled-air spectrum:

```bash
python scripts/csu_inp_processing.py analysis.inptk --out csu.csv \
  --sample A --curve A --allow-missing-header
```

Selection flags can be omitted when only one sample/curve remains. Use
`--header KEY=VALUE` for CSU header fields; `--allow-missing-header` leaves
unavailable descriptive fields blank. Recorded normalization metadata cannot be
changed. The exporter preserves the saved final points and errors without
recalculating concentrations. Its columns are `degC`, `dilution`, `INPS_L`,
`lower_CI`, and `upper_CI`; the two CI columns contain **error widths**, not
interval endpoints. Points combining several dilutions leave `dilution` blank.
Outputs must be new paths outside the saved analysis folder. Both CSV exporters
accept `--table resampled` when a separate resampled table was requested.

## Work one step at a time

You can run the main calculation as separate steps and stop after any step:

```python
ranges = {"A_neat": {"min_C": -15}}
fractions = inptk.frozen_fraction(experiment)
estimated = inptk.estimate_concentration(
    fractions,
    experiment=experiment,
    curves={"A": {"inputs": ["A_neat", "A_diluted"], "cycle": "1"}},
    method="mle",
    temperature_ranges_C=ranges,
)
converted = inptk.convert_concentration(estimated, experiment.samples, basis="sampled_air")
final = inptk.finalize_spectrum(converted, decrease_policy="stop_at_decrease")
# Optional display/export grid, after all calculations and final selection:
sampled = inptk.resample_spectrum(final, step_C=0.5, method="sample")
```

Frozen fractions need only the labelled counts table. Concentration calculations
also need `experiment`, which supplies each measurement's dilution and droplet
volume. Combining receives the frozen-fraction table because it needs the frozen
and total droplet counts stored there. Those counts supply both concentration
estimates and their uncertainty. Each step returns one scientific table.
`table.history` records its processing steps; `table.warnings` reports unavailable
results or settings that could not apply.

To calculate separate spectra for all physical inputs, use `cumulative_spectrum`
with the same fractions, experiment, and temperature ranges.
For optional differential output, call `differential_spectrum` with the same
fractions, experiment, and temperature ranges.
`inptk.subtract_blanks(estimated, {"A": blank_spectrum})` subtracts a calculated
sample/filter blank spectrum before unit conversion. `finalize_spectrum`
then selects the final nondecreasing concentration rows without changing the
input table, concentrations, or error bounds. It returns one spectrum table.
The complete workflow places retained points in each curve's `cumulative` table
and excluded points in its `excluded` table, including the exclusion reason.
It saves these quantities together with original observations and analysis choices.

Differential concentration is the increase between adjacent cooling observations,
divided by their actual temperature difference. Repeated-temperature and warming
transitions have no cooling interval; their source identities are recorded in
history, and the calculation never skips across them. Each
endpoint uses its own concentration estimate, so changing blank-corrected totals
are supported. Missing or excluded endpoints produce a missing interval; the
calculation does not bridge gaps. Negative changes carry quality flag `2`;
non-finite results carry flag `1` (flags can combine). Each row stores both interval edges;
there is no invented interval before the first supplied state. This calculation
does not establish whether a change caused by lost droplets is unbiased.

## Estimate concentration within explicit temperature ranges

Use `method="mle"` (default) or `method="average"` with the same temperature ranges:

- **MLE**, maximum likelihood estimation, finds the concentration most consistent
  with the eligible measurements' frozen and total droplet counts together.
- **Average** calculates each eligible measurement's concentration in the original
  suspension, then takes their equal-weight arithmetic mean.

By default, measurements of the same original sample, run, and cycle form one
output curve. Explicit curves can combine independent sets from different runs;
each run keeps its own blank background. With one eligible measurement, both
methods use that measurement's concentration and uncertainty. With none, neither
invents a value. There is no separate stitching operation.

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
- Limits apply to both the calculation temperature and the original observation
  selected to supply it. They follow a measurement through its cycles. They do not delete source observations or subtract frozen
  events from the remaining cumulative counts. Unknown IDs and reversed limits
  raise errors.

No hidden minimum-unfrozen-droplet cutoff or automatic range selection is applied.
Future tools may suggest ranges, but should return this same explicit mapping for
review. To use different ranges for different cycles, analyze those cycles in
separate jobs.

The same `temperature_ranges_C` keyword is accepted by `cumulative_spectrum`,
`estimate_concentration`, and `differential_spectrum`. Individual concentration rows
outside a range retain their count columns but have missing concentration/error
values and `selection_status="outside_temperature_range"`. The frozen-fraction
table remains complete. Missing blank coverage outside a sample's selected range
does not block analysis; eligible temperatures still require blank coverage.

The count model assumes independent physical droplet sets across measurements.
Each temperature is estimated separately: seeing the same droplets at another
temperature or in another freezing cycle does not create extra independent
observations. A curve cannot contain different cycles of the same run.

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

After concentration estimation, any requested blank subtraction, and unit conversion,
`decrease_policy` selects concentration rows independently for each output curve.
It follows observation order and compares each finite concentration with the last
retained value. Native observations use chronological `time_s` with stable ties,
or retained input order when time is absent. Curves requiring alignment use their
explicit warm-to-cold target order:

- `"stop_at_decrease"` (default): at the first lower concentration beyond the
  numerical-equality margin below, exclude that point and every later point,
  even if the curve later recovers.
- `"skip_decreases"`: exclude a lower point beyond that margin and continue
  checking later points. Retain them if they equal or exceed the last retained
  value within numerical precision.

For concentrations `10, 12, 11, 13` in observation order, the default retains `10, 12`;
`skip_decreases` retains `10, 12, 13`. Equality allows a fixed relative numerical
margin of `1e-9`, with zero absolute margin, to avoid treating solver roundoff as
a decrease. Outside this tiny margin, a decrease triggers the rule. This is not
a user setting or a smoothing window; values and error bounds stay unchanged.
The numerical margin is recorded in selection history.
Nonfinite values are excluded and do not establish a comparison value. Negative
finite concentrations are not clipped; this selection rule alone does not
establish that they are scientifically usable.

Neither policy raises a concentration to make a plateau or changes error bounds.
Each curve's `cumulative` table contains retained rows. Its `excluded` table contains
excluded rows after blank correction and unit conversion, with `used_in_final` and
`final_selection_status` to explain selection. Statuses are `kept`, `nonfinite`,
`decrease`, or `after_decrease`. `segment_id` marks uninterrupted retained portions
so later resampling cannot bridge exclusions. Original observations remain available.
This final selection is separate from alignment and measurement temperature ranges.
It does not sort native observations by temperature or trim the record at its
first coldest observation.

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
Output-grid, output-unit, uncertainty, and final-decrease settings remain
workflow arguments. Use `--decrease-policy stop_at_decrease` (default) or
`--decrease-policy skip_decreases`.

Use `--sample A --cycle 01` to analyze only those exact labels; repeat either
option to select several. Sample selection uses the original sample name, while
temperature ranges use measurement names. Unknown selections raise an error.
Selection and retained source metadata are recorded in the saved experiment.
`--format saved` reuses the original experiment, not the previous analysis settings;
supply the desired method, groups, ranges, and other options again.

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

Sample and cycle observations stay in labelled table rows. `observation_id`
identifies one count observation within its measurement/run/cycle; repeated
observed temperatures are valid. Missing observation IDs are generated once on
import. `picture_id`, when supplied, identifies the image and is separate from the
row identity. Instrument rows without an image remain observations.

Combined curves use `sample_id`, `curve_id`, and `point_id`. They do not invent a
single run or cycle identity when several runs contribute. Each point records its
actual source measurement, run, cycle, observation, temperature, and optional time.

The default curves keep each sample/run/cycle separate. To combine independent
inputs from different runs with different cycle labels, specify each input's
measurement/cycle pair explicitly:

```python
curves = {
    "A_replicates": {"inputs": [
        {"measurement_id": "A_run1_neat", "cycle_id": "1"},
        {"measurement_id": "A_run1_diluted", "cycle_id": "1"},
        {"measurement_id": "A_run2_neat", "cycle_id": "2"},
    ]}
}
result = inptk.analyze_concentration(experiment, curves=curves)
selected = result.curves["A_replicates"].cumulative
```

All members must represent the same original sample, and each group can contain
only one cycle from a given run. Different runs may choose different cycle labels.
Run identity comes from measurement metadata. Duplicate members, unknown IDs,
blank members, and groups mixing parent samples are rejected. An explicit mapping
requests only those groups. Repeated freezing of the same droplets never becomes
additional independent droplets. The same keyword is accepted by
`estimate_concentration`; the CLI accepts this mapping as JSON or a file through
`--curves`.

Named curves request only their selected input/cycle members. Their required
blank observations are retained for calculation. An unrequested run cannot block
a fit because it lacks blank coverage. The saved `Experiment`, `counts` and
`frozen_fraction` retain the full original observations.

Lists also hold selections, warnings, and ordered processing history. Selecting
one or many labels returns the same table type. `to_dataframe()` returns a copy,
so editing it does not change the stored result.

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
provided when available so calculations follow acquisition order. Native input
retains temperature holds, small reversals, and subsequent warming; callers must
choose the experimental observations and cycles they intend to analyze.

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
Different original samples may use different blank groups. Matching cycles and
observed temperature coverage are required; exact measured temperatures may differ
because latest-warmer alignment is automatic. Cycles are never pooled.

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
**one shared background per unit liquid volume for each contributing run and
selected cycle** at each calculation temperature. It assumes
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
counts with each run's own blank group; average uses their individually estimated
concentrations. Cross-run MLE estimates one original-sample concentration while
allowing a separate background for each run. In an MLE fit, each independent blank set contributes once at a temperature,
regardless of how many dilutions reference it. Distinct droplet volumes remain
in the probability model; unequal-volume blank counts are not collapsed into
one frozen fraction. Repeated cycles remain separate.

A blank with 32 wells is not subtracted as 32 droplets from a sample with four
wells. Each count is interpreted with its own observed total and droplet volume.
At the same frozen fraction and volume, more blank wells imply the same background
level, with greater statistical precision. The joint fit uses that stronger
information; the number of wells is not a multiplier for the correction.

Temperature ranges select whole count observations; there are no count-rebasing
or arbitrary contribution-weight settings. This model also differs from the
optional Python `blank_by_curve` operation, which subtracts an already calculated
sample/filter blank spectrum later in the workflow.

### Turn water correction on or off

Water correction is optional. With a map present, the default
`water_blank_correction=True` enables it. To analyze only the sample observations:

```python
result = inptk.analyze_concentration(experiment, water_blank_correction=False)
```

The same boolean is accepted by `cumulative_spectrum`, `estimate_concentration`, and
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

### Calling the CLI from an interactive application

Use an argument list with a separate process. These commands support GUI setup,
plot previews and calculation without importing INP-toolkit into the GUI:

```sh
inptk capabilities
inptk preview counts.csv --format native --json
inptk analyze counts.csv --metadata measurements.csv --out preview.inptk --json
inptk export-csv preview.inptk --table cumulative --out final.csv --json
```

`capabilities` returns the installed version, saved-format version and actual CLI
flags, defaults and choices as JSON. Other commands accept `--json` before or
after the command name. They return one JSON object on standard output, including
argument and input errors. Human-readable `--help` and `--version` remain unchanged.

Responses contain `protocol_version: 2`, `toolkit_version`, `saved_format_version`,
`command`, `status` (`ok` or `error`) and `warnings`. On success, `analyze` reports
the absolute saved-result path, reusable settings, original-table row counts,
and a `curves` dictionary containing names, sources, kinds and quantity row counts. `preview`
returns a frozen-fraction table with original counts, temperatures and identities,
measurement summaries, available metadata and missing suspension metadata. It
performs no fitting, correction or temperature selection and writes no result
files. Its `suspension_metadata.valid` checks metadata only; it does not validate
chosen blank coverage, groups, ranges or air/soil normalization.

Failures contain `error.code` and `error.message`. Exit codes are 0 for success,
1 for input/processing failure, 2 for invalid arguments and 130 for an interrupt.
Unexpected errors retain a diagnostic traceback on standard error. A terminated
or crashed process may have no complete JSON response; use its exit status and
keep the previous successful result. Always use a new path for each calculation.
Nonfinite numbers use the same `{"$nonfinite":"inf"}` encoding as saved files.

`--sample-map`, `--water-blank-map`, `--temperature-ranges` and
`--curves` accept a JSON object directly or a JSON file. Pass them as
individual arguments; do not construct a shell command from GUI text. See
[ICESCOPY_HANDOFF.md](ICESCOPY_HANDOFF.md) for the process and plotting contract.

## Results and scientific methods

`AnalysisResult` keeps the original `experiment`, `settings`, `history`, and
`warnings`. Counts and observed fractions belong to physical inputs. Concentration
outputs belong to named curves; no combined droplet counts or fractions are invented.

| Attribute | Type | Meaning |
|---|---|---|
| `result.counts` | `CountsTable` | Original sample and blank observations |
| `result.frozen_fraction` | `FrozenFractionTable` | Observed fractions added to those original counts |
| `result.curves[name]` | `CurveResult` | One requested output, individual or combined |
| `curve.cumulative` | `CurveSpectrumTable` | Retained native concentration points and uncertainty |
| `curve.sources` | List of records | Selected physical inputs, cycles, dilution, volume and blank assignments |
| `curve.excluded` | `CurveSpectrumTable` | Excluded native points with selection reasons |
| `curve.resampled` | `CurveSpectrumTable` or `None` | Optional temperature-grid view |
| `curve.differential` | `DifferentialSpectrumTable` or `None` | Optional activity per degree for an individual curve |

Curve tables record `contributing_measurement_ids`, `contributor_count`, and
`selection_status` (`single`, `combined`, or `no_eligible_measurements`).
`available_measurement_ids` lists measurements covering a target before ranges;
`source_measurement_ids` lists the group's members. ID lists are JSON strings.
`source_measurement_id` is populated only for a single contributor.
`source_observations` records the physical source rows, their roles, observed
rather than target temperatures, and exact/latest alignment. `point_order` is the
calculation order. These fields explain missing data without treating them as zero.
Differential spectra are optional. In the complete workflow they currently require
one input per curve, suspension units and no additional sample/filter blank spectrum.
They include only intervals whose two original observations were retained; gaps
are not bridged. For individual input calculations outside the complete workflow,
use `differential_spectrum`. Combined differential spectra are not implemented.

### Original temperatures, alignment, and optional output grids

`frozen_fraction(experiment)` preserves every source count row and temperature.
One sample measurement keeps its native observation sequence. Measurements from
the same run with matching time/temperature sequences also stay at those native
states, including repeated temperatures and holds. Without time, a matching
sequence of unique temperatures is enough to establish this correspondence;
generated row numbers alone are not evidence of simultaneous observations.

Only unsynchronized observations or cross-run groups require alignment. Their
targets are the union of the original sample temperatures, ordered warm to cold.
At each target, use the latest eligible observation at that temperature or warmer.
There is no allowance for colder observations and no interpolation of counts.
Both source and target must satisfy the measurement's range and lie within its
observed support. A sample and its blank use a matching acquisition when available, even if another
sample is missing rows. If members request different acquisitions of one shared
blank, use that blank's latest-warmer state at the target once and record alignment.
Otherwise the blank uses the same latest-warmer rule. A missing eligible blank is an error.
A native sample keeps its own temperatures even when its blank needs alignment.

The workflow does not round temperatures, add warm zero counts, select maxima,
or cut observations off at the first temperature minimum. There are no public
`temperature_method`, `temperature_tolerance_C`, or pre-calculation `step_C`
arguments. Original counts stay available in the saved experiment.

To request a regular grid **after** calculation and final selection:

```python
result = inptk.analyze_concentration(
    experiment,
    output_step_C=0.5,
    output_method="sample",  # or "interpolate"
)
result.to_dataframe(table="resampled")
result.export_csv("grid.csv", table="resampled")
```

`sample` takes the latest retained observation at or warmer than each grid target,
including its unchanged uncertainty. `interpolate` first uses that same rule at
each observed temperature to form ordered-temperature endpoints, then linearly
connects their concentrations and lower/upper interval endpoints. This prevents
small temperature reversals from placing an older, lower concentration after a
newer value. It does not change or refit any original observation.

Both choices record source point IDs, actual source temperatures, and sampling
status. `endpoint_temperatures_C` separately records the temperatures used to
construct interpolation endpoints; these can differ from source temperatures.
Interpolated bounds are not newly fitted confidence intervals. Neither method
adds extrapolation beyond a retained segment or bridges excluded observations;
any existing source extrapolation flags remain visible. Native `curve.cumulative` remains
unchanged; the optional grid is a separate `resampled` table.

The CLI equivalents are `--output-step-C 0.5 --output-method sample`, followed by
`inptk export-csv analysis.inptk --table resampled --out grid.csv`.

Concentration columns are `concentration`, `unit`, and `basis`. `lower_error`
and `upper_error` are error-bar widths: interval endpoints are concentration
minus/plus these widths. Their calculation method is recorded. They do not
include variability across repeated cycles. Non-finite estimates remain in each
curve's `excluded` table with their quality flags and are excluded from `cumulative`;
inspect flags before plotting or further analysis.

Optional `blank_by_curve={"A": blank_spectrum}` maps a named output curve
to one already calculated sample/filter blank curve. Match by exact temperature;
the blank must have one value per temperature and cover all finite target points.
Sample that blank explicitly if needed. Missing coverage is an error. Its separate,
approximate error propagation assumes independent sample and blank errors.
Corrected values may be negative and are not clipped. Final selection follows
correction and unit conversion, retaining all candidates for inspection.

`output_basis` is `suspension`, `sampled_air`, or `dry_soil`. Changing this basis
returns the same cumulative table type. Converting an already normalized result
again is rejected.

## Python file structure

```text
src/inptk/
  __init__.py       public imports
  tables.py         scientific table types and identity checks
  experiment.py     sample/measurement information, Experiment, AnalysisResult, CurveResult
  readers.py        native CSV/Python and Icescopy import
  methods.py        method, named-curve inputs, and temperature-range validation
  processing.py     native count-to-fraction and per-measurement spectrum steps
  alignment.py      match original sample/blank observations only where needed
  resampling.py     optional grid views after final concentration selection
  workflows.py      concentration estimation, correction, units, and full workflow
  results.py        assemble named curves with their sources and exclusions
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

Saved analyses use format version 3 in `analysis.json`; other versions are rejected. They include the writing
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
