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
must not alter a saved analysis. Keep ordinary controls visible and place the
remaining controls under **Advanced**; changing the dilution method shows only
its relevant controls.

| Setting | Process dialog | Preferences |
| --- | --- | --- |
| `dilution_method` | Automatic stitching, manual stitching, or MLE (fitting dilution counts together) | Optional initial choice; default automatic |
| `output_basis` | Suspension, sampled air, or dry soil; require the corresponding metadata | Do not assume one basis for every sample |
| `step_C` | Output temperature spacing | Default 0.5°C |
| `temperature_method` | Advanced observation-selection override | Default `latest` |
| `temperature_tolerance_C` | Advanced temperature allowance | Default 0°C for `latest`; `window_max_count` uses 0.01°C and `max` uses 0.05°C when omitted |
| `z` | Advanced uncertainty setting | Default 1.96, for nominal 95% bounds |
| `differential` | Request additional concentration-per-degree output | Optional starting checkbox; default off |
| `water_blank_map` | Assign one or more raw water-blank droplet sets to each sample measurement | Never store measurement names globally |
| `water_blank_correction` | Visible Apply water-blank correction checkbox; retain assignments when off | Initial value on; no map means no correction is applied |
| `blank_by_sample` | Separate subtraction of a calculated sample/filter blank spectrum; currently Python-only | Never store sample names globally |
| `decrease_policy` | Visible final-curve choice: stop at first decrease, or skip decreases and allow recovery | Default `stop_at_decrease`; allow per-analysis override |
| `min_unfrozen` | Automatic stitching: minimum unfrozen droplets | Default 3 |
| `switch_temperatures_C` | Manual stitching: draggable switches and numerical values | Never store run-specific switches globally |
| `temperature_eligibility_C` | MLE: warm cutoff for each measurement name | Never store measurement cutoffs globally |
| `mask_mode` | Raw-blank MLE uses `drop_rows`; the older corrected-input route also permits explicit baseline removal | Do not silently reuse a baseline interpretation |
| `likelihood_weights` | MLE without raw water correction; unavailable while correction is enabled | No global measurement map |
| `action_counts` | MLE without raw water correction; unavailable while correction is enabled | No global measurement map |
| `action_weight_lambda` | MLE without raw water correction; unavailable while correction is enabled | No universal scientific default |
| `action_weight_half_life` | MLE without raw water correction; unavailable while correction is enabled | No universal scientific default |
| `confidence_drop` | Expert MLE uncertainty override; normally use the value derived from `z` | Avoid a second competing uncertainty default |

Direct weights and action-based weights are alternatives. Decay rate and half-life
are alternatives. All measurement maps use exact measurement names, not dilution
factors. For Icescopy these are exported `sample_name` values such as `Sample_2`.
A parent sample groups those measurements; it is a different identity.

Sample grouping, measurement/run/cycle names, dilution factors, droplet volumes,
air/suspension/soil amounts, and blank assignments are analysis data. Keep them in
the session and analysis, not application Preferences. Never silently assume
50 µL, a dilution factor, an air volume, or that similarly named measurements
belong to the same original sample.

## Process dialog and plots

Temperature import attaches temperatures and times to observations. The new INP
processing dialog owns water-blank assignment, metadata review, dilution grouping,
stitching, MLE exclusions, and final-curve selection. Preserve raw session counts
before any correction; do not make temperature import irreversibly subtract a
blank or choose the INP analysis settings.

- Select the parent sample, run, and cycle. Preserve every cycle separately; do
  not pool repeated freezing cycles as independent droplets.
- Make water-blank correction optional with a visible checkbox. Turning it off
  keeps raw blank observations and assignments but excludes them from calculated
  sample tables. Show the returned `water_blank_correction_applied` state; an
  enabled checkbox without a map does not mean correction occurred.
- Select the water-blank droplet sets explicitly, with one or more names per
  sample measurement. Show their measured counts and individual curves. Require
  each set's known droplet volume, which may differ between samples and blanks.
  State the supported common background per volume and same prepared-water
  protocol beside this assignment; there is no automatic protocol detection.
- Use plots as the main view: counts, frozen fractions, individual dilution
  concentrations, combined concentration, and optional differential results.
  Keep tables available for inspection and export.
- Every temperature axis increases **left to right**, for example **−30°C → −5°C**.
  Plot sorting must not change the calculation order or saved observations.
- Keep measurement colors consistent. Label curves with measurement name and
  dilution factor. Show excluded observations faintly instead of deleting them.
- Automatic stitching has only the minimum-unfrozen-droplet setting. It keeps
  the current dilution through its coldest eligible point, then switches to the
  next dilution for colder temperatures. Gaps do not trigger early switching.
  Each selected concentration and its error bounds remain unchanged; there is no
  overlap setting, averaging, or refitting between curves.
- For manual stitching, display draggable switch lines and numerical entries.
  Highlight the selected curve segments and show the combined preview.
- For MLE, display a warm cutoff for each measurement and the fitted curve.
  A −15°C cutoff retains **T ≤ −15°C**, on the colder left side. The excluded
  warmer observations are on the right.
- Show the final-decrease choice beside the combined plot. Use `final_candidates`
  to draw excluded points faintly and explain each selection flag; overlay the
  retained `final` curve. Do not draw an invented plateau or hide excluded data.
- Distinguish edited controls from the last calculated result. Provide
  **Recalculate**, **Cancel**, and **Save**; display package warnings and errors.

The current MLE control supports one warm cutoff per measurement. It does not
support arbitrary excluded intervals or individual-point exclusions. Do not offer
an unrestricted exclusion brush. `drop_rows` omits warmer sample rows from fitting
and is the supported cutoff mode while raw water correction is enabled. On the older corrected-input
route, `rebase_counts` additionally subtracts the warm frozen baseline and its
droplets. Do not expose that operation or contribution weights in raw-blank mode.

Use one CLI job per selected sample/cycle when switches or cutoffs differ between
groups. Manual stitching requires the same dilution-factor set in all groups
supplied to one job and one switch list for that job. Its list is ordered **warm
to cold**, e.g. `[-12, -16, -20]`, even though the plot axis increases left to right.

Final selection runs after blank correction and unit conversion, independently
for each sample/run/cycle, examining finite values from warm to cold:

- `stop_at_decrease` stops at the first value below the last retained value and
  excludes every colder point. Even a small decrease triggers this; there is no
  hidden tolerance or multi-point window.
- `skip_decreases` excludes a lower value but keeps checking colder points,
  retaining values that recover to or exceed the last retained value.

Equal concentrations are retained. Nonfinite values are excluded and do not set
the comparison value. Negative values are not clipped. Neither choice changes
concentrations or error bounds. This final selection does not change which raw
observations were selected or which measurements entered an MLE fit.

## CLI and saved-result contract

Use Qt's separate-process support with an executable path and argument list;
do not build a shell command. For new analyses, export **raw sample and raw blank
counts** to native long CSV from the Icescopy session. The existing
`build_freeze_count_timeseries_csv_text` output may already be corrected and is
not the raw contract. Preserve it as an existing result; do not reverse its
correction to reconstruct observations.

`raw-counts.csv` needs `measurement_id`, `cycle_id`, `temperature_C`, `n_total`,
and `n_frozen`, plus `time_s` when available. Each physical sample or blank droplet
set keeps its own name. Do not duplicate a blank's rows for each sample that uses
it. Provide one metadata row per set, for example:

```csv
measurement_id,sample_id,run_id,dilution,droplet_volume_uL,sample_type,air_volume_L,suspension_volume_mL,filter_fraction_used
Sample_0,Sample_A,run-01,1,50,air,100,5,1
Sample_1,Sample_A,run-01,10,50,air,100,5,1
Water_1,water1,run-01,1,20,other,,,
Water_2,water2,run-01,1,50,other,,,
```

These volumes are examples; use the actual recorded values. Native input groups
sample measurements through `sample_id` in this metadata, without `--sample-map`.
Blank sets must have dilution 1 and the same run as the assigned sample sets.
Save `water-blank-map.json` as a mapping to lists, including single-blank lists:

```json
{"Sample_0": ["Water_1", "Water_2"], "Sample_1": ["Water_1", "Water_2"]}
```

Assign every nonblank measurement. Dilutions of one original sample/run must use
the same set of blanks; their order in the list is irrelevant. Different original
samples can choose different groups. Counts and droplet volumes can differ across
physical sets. Each sample/blank assignment needs matching cycle and temperature
observations. Repeated cycles never become extra independent blank droplets.

An MLE `method-options.json` can contain:

```json
{"temperature_eligibility_C": {"Sample_1": -15}, "mask_mode": "drop_rows"}
```

```sh
inptk analyze raw-counts.csv --format native --metadata measurements.csv \
  --water-blank-map water-blank-map.json --sample Sample_A --cycle 0 \
  --dilution-method mle --method-options method-options.json \
  --output-basis sampled_air --step-C 0.5 \
  --temperature-method latest --temperature-tolerance-C 0 \
  --decrease-policy stop_at_decrease --out analysis.inptk
```

`--sample` selects exact parent sample IDs; `--cycle` selects exact cycle IDs.
Both can be repeated. Selection retains associated blank counts once per physical
set and selected run/cycle, together with required metadata. Blank-only parent
samples cannot be selected as analysis samples. The saved `source.selection`
records the user selection, not the added blank context. `--run-id` labels imports;
it is not a run filter for saved input. Unknown selections fail.

For manual stitching, use `--dilution-method manual` and, for three dilution levels:

```json
{"switch_temperatures_C": [-12, -16]}
```

Saved experiments retain `water_blank_map`; `--format saved` uses it directly and
rejects a replacement map. The existing `--format icescopy` route still accepts
corrected exports with an explicit `--sample-map` JSON object, but rejects
`--water-blank-map`. Its metadata overrides use Icescopy measurement labels as
`sample_id` and `well_volume_uL` for volume, for example
`[{"sample_id":"Sample_0","well_volume_uL":50}]`. Native metadata instead uses
`measurement_id` for the physical set and `droplet_volume_uL` for its volume.

Add `--no-water-blank-correction` to analyze the sample counts without correction.
The saved original experiment still includes its raw blanks and assignment map;
calculated tables exclude blank sets, and blank temperature coverage is not
required. The same choice is `water_blank_correction=False` in the full Python
workflow and its cumulative, differential, and dilution-combination steps. With
no blank map, both settings use the ordinary sample-only calculation. Neither
choice reverses correction already present in an imported count table.

Read the result only after the process exits successfully. Capture standard output,
standard error, and exit status; report failures without replacing a previous
successful result. Exit status 0 means success, 1 a handled input/processing
failure, and 2 an argument-parsing error. Each output directory must be new. Saving refuses overwrites.
Remove disposable previews when they are no longer needed.

`analysis.inptk` is a directory containing `analysis.json`, not a binary archive.
Check `format == "inptk"` and `format_version == 1`. The payload records
`toolkit_version`, original counts, each set's volume metadata, `water_blank_map`,
result tables, resolved settings, history, and warnings. Settings include requested
`water_blank_correction` and effective `water_blank_correction_applied`. Tables contain columns and row records. Nonfinite values
use explicit objects such as `{"$nonfinite": "inf"}`; decode them for display,
retain their quality flags, and do not draw them as finite concentrations.

Read `frozen_fraction`, `per_dilution`, `combined`, `final_candidates`, `final`, and
optional `differential` tables for plotting. `final_candidates` contains all rows
after blank correction and unit conversion, with `used_in_final` and
`final_selection_status` (`kept`, `nonfinite`, `decrease`, or
`colder_than_decrease`). It is optional when reading older saved analyses;
new full-workflow results include it. `final` contains only retained rows.
`lower_error` and `upper_error` are widths, so
bounds are concentration minus/plus those widths. They are not interval endpoints.
`source_measurement_ids` lists candidate measurements, not proof that every one
contributed at every temperature. Automatic and manual stitching identify each
selected point with `source_measurement_id`; an empty value means unavailable.
Each point uses one measurement, with its concentration and bounds unchanged.
Manual results additionally report `selection_status`.

Save the input identity, executable/version, requested options, and returned
resolved settings with the Icescopy analysis. Do not parse human-readable progress
messages as a data format. The current CLI has no structured capability/status
command and no stage-only command for a counts/fraction preview. The concentration
workflow requires dilution and droplet-volume metadata; requested air/soil units
require their additional metadata. Zero-total rows are rejected, not discarded.
Any exclusion of invalid source rows must be an explicit, recorded decision.

## Scientific boundaries to retain

- Raw-blank analysis assumes one common background concentration per volume at
  each temperature for the assigned sets, using the same prepared-water protocol.
  Its Poisson model describes randomly distributed ice-active contributions per
  volume. It does not model a separate well-surface-area contribution or establish
  that different preparation protocols share a background.
- Each sample and blank set retains its own positive known droplet volume and
  observed count total. Unequal volumes enter separate probabilities; do not pool
  unequal-volume counts into one frozen fraction for calculation.
- Individual dilution spectra and MLE use the same joint sample/blank model.
  Joint fitting means estimating sample concentration and background from their
  counts together. The common blank group contributes once per temperature to a
  multi-dilution MLE fit. Stitching copies one dilution's fitted concentration and
  uncertainty bounds at each temperature without combining the uncertainty from
  multiple curves or fitting an overlap region.
- While raw water correction is enabled, use `latest` or `max` temperature
  selection. `window_max_count` is rejected because its synthetic warm zero rows
  are not observed droplets. Its legacy behavior remains available with correction
  disabled or no blank map.
- Raw-blank MLE accepts `drop_rows` temperature cutoffs and rejects `rebase_counts`,
  direct likelihood weights, and action-based weights. Confidence settings remain
  explicit; background uncertainty must be included when calculating bounds.
- Existing corrected Icescopy exports retain their earlier approximate
  count-based processing. Those adjusted counts alone cannot recover the measured
  blank's uncertainty or the dependence introduced by a shared blank. Do not
  subtract again or reconstruct supposedly raw counts from an inverse correction.
- MLE fits temperatures separately; observations of the same droplets at other
  temperatures are not additional independent droplets. Cycles also remain
  separate. There is no public `enforce_monotone` option; final-decrease selection
  excludes rows without fitting them again or changing their uncertainty.
- Differential output is per measurement: the change in adjacent cumulative
  concentrations divided by the actual cooling-temperature interval. The first
  point has no preceding interval and is omitted. Corrected totals may vary;
  negative or nonfinite results are flagged. Unexplained loss of droplets is a
  separate scientific problem, not automatically a valid background correction.
- Missing or invalid inputs must remain visible. Do not silently guess metadata,
  fill excluded ranges from another measurement, or hide a corrected downturn.

## Implementation locations

In the **Icescopy repository**:

- `src/icescopy_aux.py`: `PreferencesDialog`, defaults, category registration,
  and the existing ML page's Browse layout.
- `src/icescopy_paths.py`: user-preferences location and atomic saving.
- `src/Icescopy.py`: loading/applying preferences and registering analysis actions.
- `src/icescopy_session_io.py`: session persistence and the existing corrected
  CSV export; add a separate raw native export for the new INP dialog.
- `src/icescopy_plot.py`: existing PyQtGraph integration.
- `resources/preferences.xml`: bundled initial preference values.

Keep the new process runner and processing dialog in focused modules rather than
putting their calculation/control logic into the main window.

In **INP-toolkit**: `src/inptk/cli.py` defines arguments and selection,
`methods.py` validates method settings, `workflows.py` applies them, and `io.py`
defines the saved-result format. GUI integration must follow these public data and
CLI contracts rather than the private `_engine` modules.

## Verification before handoff

- 305 automated tests passed, with one existing pandas warning in the deliberately
  invalid-timestamp test.
- Ruff checks, mypy checks of 19 source files, and Bandit checks passed.
- Verification includes numerical and regression checks for raw sample/blank
  fitting, known unequal volumes, optional correction, physical blank assignments,
  separate cycles, CLI and saved-result parity, final-point selection, and copying
  one dilution's concentration and bounds per stitched point.
- These checks verify the implemented calculations and software behavior. They
  do not establish uncertainty-interval coverage across all experimental
  conditions or verify that a particular set of experimental blanks shares the
  assumed common background.
- Notebook execution and installed-CLI checks are pending completion. The old
  corrected-input notebook is not validation of the new raw-blank model.
- No Icescopy GUI files have been changed; its integration remains to be
  implemented and tested.
