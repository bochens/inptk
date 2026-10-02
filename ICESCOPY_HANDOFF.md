# Icescopy integration handoff

INP-toolkit 0.2.0 performs the calculations. Icescopy provides the controls and
plots, launches the separately installed `inptk` executable, and reads the saved
results. Do not import INP-toolkit into Icescopy or copy its scientific calculations.
No Icescopy GUI changes are included in this release.

## Preferences and analysis settings

Add **Preferences → INP toolkit** with an executable chooser, **Browse**, **Test
connection**, detected version, and connection status. Test with `inptk --version`.
Use the existing user-preferences location and save mechanism.

Preferences hold starting values for new analyses. The process dialog shows the
values used for this analysis and saves them with the result. Changing Preferences
must not alter a saved analysis. Offer two combination choices: MLE (maximum
likelihood estimation, fitting the eligible counts together) and Average (the
equal-weight mean of eligible per-measurement concentrations). MLE is the default.
Average uses conservative bounds that allow shared blank uncertainty; MLE uses
counts jointly. Keep this explanation beside the method choice.

| Setting | Process dialog | Preferences |
| --- | --- | --- |
| `method` | MLE or Average, using the same ranges | Default `mle`; allow per-analysis override |
| `temperature_ranges_C` | Cold and warm limits for each exact measurement ID; both endpoints included | Never store measurement-specific ranges globally |
| `water_blank_map` | Assign one or more raw water-blank sets to each sample measurement | Never store measurement names globally |
| `water_blank_correction` | Visible Apply blank correction checkbox; retain assignments when off | Initial value on; without a map, no correction is applied |
| `output_basis` | Suspension, sampled air, or dry soil; require the corresponding metadata | Do not assume one basis for every sample |
| `step_C` | Output temperature spacing; the only observation-matching control | Default 0.5°C |
| `temperature_method` | Fixed to `latest` by the integration; save explicitly, without a GUI control | No preference control |
| `temperature_tolerance_C` | Fixed to 0°C by the integration; save explicitly, without a GUI control | No preference control |
| `z` | Advanced uncertainty setting | Default 1.96, for nominal 95% bounds |
| `differential` | Request additional concentration-per-degree output | Optional starting checkbox; default off |
| `decrease_policy` | Visible final-curve choice: stop at first decrease, or skip decreases and allow recovery | Default `stop_at_decrease`; allow per-analysis override |

For normal GUI use, automatically take the latest observation at the requested
temperature or warmer. Only grid spacing needs a temperature-matching control.
Python and CLI users can still choose `max` or a nonzero tolerance explicitly;
those advanced capabilities do not need controls in the initial Icescopy dialog.

The separate Python-only `blank_by_sample` argument subtracts an already calculated
sample/filter blank spectrum. It is not a raw water-blank assignment and has no
CLI control in this release.

Sample grouping, measurement/run/cycle names, dilution factors, droplet volumes,
air/suspension/soil amounts, and blank assignments are analysis data. Keep them in
the session and analysis, not Preferences. Never assume a droplet volume, a
dilution factor, an air volume, or that similarly named measurements belong to
the same original sample. Measurement maps use exact names, such as Icescopy's
`Sample_2`, not dilution factors or parent sample names.

## Process dialog and plots

Temperature import attaches temperatures and times to observations. The new INP
processing dialog owns sample grouping, metadata review, blank selection,
temperature ranges, and final-curve selection. Remove the old temperature-import
blank picker when this dialog is connected. Existing saved assignments can
prefill the new controls. Preserve raw counts before any correction.

Keep the current order: select sample and blank observations on the requested
temperature grid, then estimate concentrations. With `latest` and zero tolerance,
a missing exact temperature uses the latest warmer observation automatically.
The saved `Experiment` retains original temperatures and counts. There is no
second interpolation or resampling of the final concentration curves.

- Select parent samples, runs, and cycles. Runs and cycles remain separate;
  repeated freezing of the same droplets must not increase the independent
  droplet count.
- Review dilution and volume metadata and group the measurements belonging to
  each original sample.
- Make blank correction optional. Select blank sets in this same dialog and show
  their counts and individual curves. State the simplifying assumption: all blank
  sources, including water, substrate, and PCR well walls, scale with liquid
  volume. Use actual sample and blank volumes; equal volumes are not required.
- Use plots as the main view: counts, frozen fractions, individual dilution
  concentrations, combined concentration, and optional differential results.
  Keep tables available for inspection and export.
- Every temperature axis increases **left to right**, for example **−30°C → −5°C**.
  Sorting for display must not change saved observations or calculation order.
- Give each measurement a consistent color and label it with its name and dilution.
  Supply a cold-limit handle (`min_C`) and warm-limit handle (`max_C`), with
  numerical entries and a clear no-limit state. Shade excluded temperatures.
- Overlapping ranges use all eligible measurements: MLE fits their counts
  together; Average takes the mean of their concentration estimates. Disjoint
  ranges restrict contribution to one measurement at a time. With one eligible
  measurement, both methods use that estimate and its uncertainty. With none,
  there is no estimate. Do not add a separate stitching control.
- Keep the full frozen-fraction curves available. Per-dilution concentration rows
  outside their ranges remain present with missing estimates and
  `selection_status="outside_temperature_range"`. Ranges do not erase input counts.
- Display the returned contributors for each combined point. There is no hidden
  minimum-unfrozen-droplet cutoff or automatic range choice. Future range
  suggestions should populate these same explicit controls for review.
- Show the final-decrease choice beside the combined plot. Draw excluded
  `final_candidates` faintly with their reasons and overlay the retained `final`
  curve. Do not invent a plateau or hide excluded observations.
- Separate edited controls from the last calculated result. Provide **Recalculate**,
  **Cancel**, and **Save**, and display package warnings and errors.

Ranges are inclusive and apply to each measurement's selected output temperatures
in all selected cycles. An omitted measurement is unrestricted. The current
contract supports one continuous range per measurement, not an arbitrary exclusion
brush or several disjoint intervals. Use separate CLI jobs when the same
measurement needs different ranges in different cycles. Missing blank coverage
outside a sample's selected range does not block calculation; missing blank
coverage at an eligible temperature is an error.

Combining different runs is not implemented. Future support would need to retain
each run's own blank observations and assignments; it must not treat another
run's blank as interchangeable.

Final selection occurs after concentration estimation, any spectrum subtraction,
and unit conversion, independently for each sample/run/cycle. It examines finite
concentrations from warm to cold:

- `stop_at_decrease` excludes the first point below the last retained value
  beyond numerical precision, and every colder point.
- `skip_decreases` excludes a lower point beyond numerical precision but keeps
  checking colder points, retaining values that recover to or exceed the last
  retained value.

Numerical equality uses a fixed relative margin of `1e-9` and zero absolute
margin, recorded in selection history. This prevents solver roundoff from
triggering a decrease; it is not a user-adjustable allowance for dips or a
smoothing window. Outside that margin, a decrease triggers the selected rule.
Nonfinite values do not set the comparison value. Negative values are not clipped.
Neither choice changes concentrations, error bounds, or the original input counts.

## CLI input and process contract

Use Qt's separate-process support with an executable path and argument list;
do not build a shell command. Call the package CLI for analysis.

For new analyses, export **raw sample and raw blank counts** to native long CSV.
The current `freeze_count_timeseries.csv` may already be corrected and does not
carry the complete raw blank context. Preserve existing
results; do not reverse their correction to reconstruct observations.

`raw-counts.csv` needs `measurement_id`, `cycle_id`, `temperature_C`, `n_total`,
and `n_frozen`, plus `time_s` when available. Each physical sample or blank droplet
set keeps its own ID. Do not duplicate a blank's rows for every sample that uses
it. Supply one metadata row per set, for example:

```csv
measurement_id,sample_id,run_id,dilution,droplet_volume_uL,sample_type,air_volume_L,suspension_volume_mL,filter_fraction_used
Sample_0,Sample_A,run-01,1,50,air,100,5,1
Sample_1,Sample_A,run-01,10,50,air,100,5,1
Water_1,water1,run-01,1,20,other,,,
Water_2,water2,run-01,1,50,other,,,
```

These are examples; use recorded volumes. Native input groups measurements through
`sample_id`, without `--sample-map`. Sample and blank sets can have different
known positive droplet volumes and count totals. Blank sets require dilution 1
and the same run as their assigned sample sets.

`water-blank-map.json` always maps to lists, even for one blank:

```json
{"Sample_0": ["Water_1", "Water_2"], "Sample_1": ["Water_1", "Water_2"]}
```

Assign every nonblank measurement when using a map. Dilutions of one original
sample/run must use the same blank set; list order is irrelevant. Different
original samples can use different groups. Matching cycle and eligible temperature
observations are required. Cycles are never pooled as extra blank droplets.

`temperature-ranges.json` contains measurement-specific limits:

```json
{"Sample_0": {"min_C": -15}, "Sample_1": {"max_C": -12}}
```

Here Sample_0 contributes at −15°C and warmer, Sample_1 at −12°C and colder, and
both contribute between −15°C and −12°C, inclusive. Omit either bound, or use
`null`, for no limit on that side. An omitted measurement uses its full range.

```sh
inptk analyze raw-counts.csv --format native --metadata measurements.csv \
  --water-blank-map water-blank-map.json --sample Sample_A --cycle 0 \
  --method mle --temperature-ranges temperature-ranges.json \
  --output-basis sampled_air --step-C 0.5 \
  --temperature-method latest --temperature-tolerance-C 0 \
  --decrease-policy stop_at_decrease --out analysis.inptk
```

`--temperature-ranges` also accepts the JSON object directly. `--method` accepts
`mle` (default) or `average`; it is the same `method=` keyword in Python. There
are no `--dilution-method` or `--method-options` arguments.

`--sample` and `--cycle` select exact parent sample and cycle IDs and can be
repeated. Selection retains required blank rows once per set/run/cycle and their
metadata. Blank-only parent samples cannot be analysis selections. The saved
`source.selection` records the requested selection, not added blank context.
`--run-id` labels imports; it does not filter saved runs. Unknown selections fail.

Saved experiments retain `water_blank_map`; `--format saved` uses it and rejects
a replacement map. The existing `--format icescopy` adapter accepts a
`--sample-map` JSON file but rejects `--water-blank-map`. Its metadata overrides
use Icescopy measurement labels as `sample_id` and `well_volume_uL` for volume,
for example `[{"sample_id":"Sample_0","well_volume_uL":50}]`. Native metadata
uses `measurement_id` and `droplet_volume_uL` instead.

`--no-water-blank-correction` analyzes sample counts without correction while
preserving raw blank observations and assignments in the saved experiment.
Calculated tables exclude blank sets, and blank temperature coverage is not
required. Show the returned `water_blank_correction_applied`: a selected checkbox
without a map does not mean correction occurred. Neither checkbox state reverses
correction already present in an imported table.

Read results only after successful process completion. Capture standard output,
standard error, and exit status. Status 0 means success, 1 a handled input or
processing failure, and 2 an argument-parsing error. Each output path must be new;
saving refuses overwrites. Keep prior successful results after a failed job and
remove disposable previews when no longer needed.

## Saved results and plots

`analysis.inptk` is a directory containing `analysis.json`. Check
`format == "inptk"` and `format_version == 1`. The payload contains
`toolkit_version`, original counts, metadata, `water_blank_map`, tables, resolved
settings, history, and warnings. Settings include `estimation_method` (`mle` or
`average`), normalized `temperature_ranges_C`, requested `water_blank_correction`, and effective
`water_blank_correction_applied`. `water_blank_model="volume_scaled"` records the
fixed physical assumption; it is not an extra user setting.

Tables contain columns and row records. Nonfinite numbers use explicit objects
such as `{"$nonfinite":"inf"}`; decode them and retain their quality flags.
`lower_error` and `upper_error` are widths: subtract/add them to concentration to
obtain the interval endpoints. Do not plot them as endpoints directly.

Read `frozen_fraction`, `per_dilution`, `combined`, `final_candidates`, `final`, and
optional `differential` for plotting. `combined` records:

- `available_measurement_ids`: measurements present before applying ranges.
- `contributing_measurement_ids` and `contributor_count`: measurements used at
  this temperature. ID lists are JSON strings.
- `source_measurement_id`: the single contributor, or an empty string otherwise.
- `selection_status`: `single`, `combined`, or `no_eligible_measurements`.
- `source_measurement_ids`: all measurements supplied to that sample/run/cycle
  group; this is not proof that every one contributed at each temperature.

With a single contributor, both methods return that measurement's concentration
and uncertainty unchanged.
Missing estimates and nonfinite fitted values carry quality flags; selection
status alone does not establish that a concentration is finite.

`final_candidates` contains rows after requested spectrum subtraction and unit
conversion, with `used_in_final` and `final_selection_status` (`kept`, `nonfinite`,
`decrease`, `colder_than_decrease`). Older saved analyses may omit this table.
New full-workflow results include it; `final` contains only retained rows.

Save input identity, executable/version, requested options, and returned settings
with the Icescopy analysis. Do not parse human-readable progress messages as data.
The CLI has no structured capability command or stage-only counts/fraction preview.
Concentration requires dilution and droplet volume; air/soil output needs its
additional metadata. Zero-total rows are rejected. Any source-row exclusion must
be explicit and recorded.

For optional CSU-formatted output, the repository helper
`scripts/csu_inp_processing.py` formats a saved `AnalysisResult`. Select one exact
sample/run/cycle, already calculated as `sampled_air` in `INP_per_L_air`:

```sh
python scripts/csu_inp_processing.py analysis.inptk --out csu.csv \
  --sample Sample_A --run run-01 --cycle 0 --allow-missing-header
```

This preserves the saved final points and error widths without refitting,
normalizing, or applying another final-selection rule. Selection flags are only
required to resolve multiple groups. `--header KEY=VALUE` supplies CSU header
fields; `--allow-missing-header` permits missing descriptive fields. Recorded
normalization metadata cannot conflict with supplied headers. The CSV columns
are `degC`, `dilution`, `INPS_L`, `lower_CI`, and `upper_CI`. The CI columns are
error widths; a combined point has blank dilution. The helper rejects existing
outputs and output paths inside the saved analysis folder.

## Scientific boundaries

- There is one background model. It combines water, substrate, PCR well walls,
  and other blank sources into a shared background per unit liquid volume. The
  simplifying assumption is that this entire contribution scales with liquid
  volume. Assigned blanks should represent the sample's assay preparation. Each
  sample and blank uses its actual volume; equal volumes are not required. There
  is no separate surface-area term or background-model selector.
- MLE explicitly fits the shared assay background. It cannot identify or remove
  contamination unique to one dilution and absent from its assigned blanks.
  Selecting measurement temperature ranges remains a user decision.
- Each sample and blank set enters with its own count total and known volume.
  A blank with 32 wells is not subtracted as 32 droplets from a sample with four
  wells. At the same frozen fraction and volume, more blank wells imply the same
  background level with greater statistical precision. Their count is not a
  multiplier for the correction. In a joint MLE fit, each
  independent blank set contributes once per temperature, even when many
  dilutions share it. Both combination methods include measured blank uncertainty
  when raw water correction is enabled.
- MLE uncertainty uses the same profile-likelihood rule for one or many sample sets.
  This means finding concentration bounds while allowing the fitted water
  background to vary. The default threshold is `z**2/2` with `z=1.96`. Supplied
  volumes/dilutions and variability between repeated cycles are not included.
- Average reports **Bonferroni-adjusted marginal profile bounds**: widen each
  measurement's profile interval for the number of contributing measurements,
  then average the lower and upper endpoints separately. The
  [Bonferroni adjustment](https://www.itl.nist.gov/div898/handbook/prc/section4/prc463.htm)
  allows dependence from a shared blank; every individual fit includes its
  assigned blank observations. Profile intervals remain approximate, so do not
  promise exact 95% coverage or smaller error bars as more measurements enter.
  One contributor retains its ordinary bounds. Infinite estimates make the
  average infinite; unavailable individual estimates make the mean unavailable,
  without silently dropping a contributor.
- Existing corrected counts cannot recover raw blank uncertainty. Their
  intervals remain conditional on those counts; never correct them again or
  infer supposedly raw counts by reversing a clipped subtraction.
- Each temperature and cycle is analyzed separately. No temperature or cycle
  increases the independent physical droplet count. There are no arbitrary
  contribution weights, count-baseline removal, or automatic range cutoffs.
- Concentration estimation accepts `latest` or `max` observation selection.
  `window_max_count` creates synthetic warm zero rows and is only a Python
  frozen-fraction preview option, not a concentration or CLI analysis option.
- Differential output is per measurement: the change between adjacent cumulative
  concentrations divided by their actual temperature interval. It has no invented
  first interval; negative or nonfinite results are flagged. Unexplained droplet
  loss requires separate scientific interpretation.

## Implementation locations

In the **Icescopy repository**:

- `src/icescopy_aux.py`: `PreferencesDialog`, defaults, categories, and the
  existing ML page's Browse layout.
- `src/icescopy_paths.py`: user-preferences location and atomic saving.
- `src/Icescopy.py`: loading/applying preferences and registering analysis actions.
- `src/icescopy_dialogs.py`: existing temperature-import blank controls and output
  dialog; move the former responsibility to the new INP process dialog.
- `src/icescopy_freeze_count_timeseries.py` and `src/icescopy_temperature_import.py`:
  raw count preparation and existing blank subtraction. Retain uncorrected counts
  before that subtraction for the new export.
- `src/icescopy_session_io.py`: session persistence and current CSV export; add the
  raw native input contract without replacing existing saved results.
- `src/icescopy_plot.py`: PyQtGraph integration.
- `resources/preferences.xml`: bundled initial preference values.

Keep the new process runner and dialog in focused modules. In **INP-toolkit**,
`src/inptk/cli.py` defines arguments and selection, `processing.py` supplies
separately callable steps, `workflows.py` applies the analysis, `water_blank.py`
prepares shared-background fits, `methods.py` validates temperature ranges, and
`io.py` defines saved results. Follow these
public data and CLI contracts, not private `_engine` imports.

## Verification before integration

Completed checks, including the approved CSU exporter:

- The full automated suite passed: 406 tests, with one existing pandas timestamp
  parsing warning. Ruff, mypy across 19 source files, Bandit, and diff checks passed.
- Wheel and source distributions were built. The installed CLI was exercised
  outside the checkout with MLE and Average, raw blanks, correction on/off,
  temperature ranges, saved-result round trips, and overwrite protection.
- The M1 notebook executed all 13 code cells and produced five figures. Source
  inputs were unchanged, and separately called processing steps agreed with the
  complete workflow.

These checks verify calculations and software behavior, not uncertainty coverage
under every experimental condition or whether particular blanks satisfy the
shared-background assumption. No Icescopy GUI files have changed; integration
and GUI testing remain separate work.
