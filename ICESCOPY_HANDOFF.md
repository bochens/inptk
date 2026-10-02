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
| `blank_by_sample` | Sample-to-blank assignment; currently Python-only, so do not enable in the CLI dialog | Never store sample names globally |
| `decrease_policy` | Visible final-curve choice: stop at first decrease, or skip decreases and allow recovery | Default `stop_at_decrease`; allow per-analysis override |
| `min_unfrozen` | Automatic stitching: minimum unfrozen droplets | Default 3 |
| `overlap_points` | Automatic stitching: points examined near a dilution transition | Default 4 |
| `switch_temperatures_C` | Manual stitching: draggable switches and numerical values | Never store run-specific switches globally |
| `temperature_eligibility_C` | MLE: warm cutoff for each measurement name | Never store measurement cutoffs globally |
| `mask_mode` | Explicit choice with MLE cutoffs: exclude warmer rows, or also remove their frozen baseline | Do not silently reuse a baseline interpretation |
| `likelihood_weights` | Advanced MLE contribution weights by measurement name | No global measurement map |
| `action_counts` | Advanced MLE handling-action counts by measurement name | No global measurement map |
| `action_weight_lambda` | Advanced decrease in weight per action; alternative to half-life | No universal scientific default |
| `action_weight_half_life` | Advanced number of actions that halves the weight; alternative to decay rate | No universal scientific default |
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

- Select the parent sample, run, and cycle. Preserve every cycle separately; do
  not pool repeated freezing cycles as independent droplets.
- Use plots as the main view: counts, frozen fractions, individual dilution
  concentrations, combined concentration, and optional differential results.
  Keep tables available for inspection and export.
- Every temperature axis increases **left to right**, for example **−30°C → −5°C**.
  Plot sorting must not change the calculation order or saved observations.
- Keep measurement colors consistent. Label curves with measurement name and
  dilution factor. Show excluded observations faintly instead of deleting them.
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
an unrestricted exclusion brush. `drop_rows` only omits warmer rows from fitting;
`rebase_counts` also subtracts the warm frozen baseline and its droplets. These
have different scientific meanings and must be labeled explicitly.

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
do not build a shell command. Export input using Icescopy's existing
`build_freeze_count_timeseries_csv_text` helper. Keep the original export unchanged.
Write sample grouping and method options as JSON files alongside the job inputs.
For example, `sample-map.json` can contain:

```json
{"Sample_0": "Sample_A", "Sample_1": "Sample_A", "Sample_2": "Sample_A"}
```

An MLE `method-options.json` can contain:

```json
{"temperature_eligibility_C": {"Sample_2": -15}, "mask_mode": "drop_rows"}
```

```sh
inptk analyze freeze_count_timeseries.csv \
  --format icescopy --sample-map sample-map.json \
  --sample Sample_A --cycle 0 \
  --dilution-method mle --method-options method-options.json \
  --output-basis sampled_air --step-C 0.5 \
  --temperature-method latest --temperature-tolerance-C 0 \
  --decrease-policy stop_at_decrease --out analysis.inptk
```

`--sample` selects exact parent sample IDs; `--cycle` selects exact cycle IDs.
Both can be repeated. `--run-id` labels imported native/Icescopy data; it is not a
run filter for a saved analysis. Unknown selections fail. For manual stitching,
use `--dilution-method manual` and, for three dilution levels:

```json
{"switch_temperatures_C": [-12, -16]}
```

Missing header metadata can be supplied with `--metadata overrides.json` using
an array of records, for example:

```json
[{"sample_id": "Sample_0", "well_volume_uL": 50}]
```

Here `sample_id` is the Icescopy measurement name. Only supplied, nonmissing
fields replace header values; the remaining header metadata is retained.

Read the result only after the process exits successfully. Capture standard output,
standard error, and exit status; report failures without replacing a previous
successful result. Exit status 0 means success, 1 a handled input/processing
failure, and 2 an argument-parsing error. Each output directory must be new. Saving refuses overwrites.
Remove disposable previews when they are no longer needed.

`analysis.inptk` is a directory containing `analysis.json`, not a binary archive.
Check `format == "inptk"` and `format_version == 1`. The payload records
`toolkit_version`, original counts and metadata, result tables, resolved settings,
history, and warnings. Tables contain columns and row records. Nonfinite values
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
contributed at every temperature. Manual results additionally report
`source_measurement_id` and `selection_status`.

Save the input identity, executable/version, requested options, and returned
resolved settings with the Icescopy analysis. Do not parse human-readable progress
messages as a data format. The current CLI has no structured capability/status
command and no stage-only command for a counts/fraction preview. The concentration
workflow requires dilution and droplet-volume metadata; requested air/soil units
require their additional metadata. Zero-total rows are rejected, not discarded.
Any exclusion of invalid source rows must be an explicit, recorded decision.

## Scientific boundaries to retain

- Icescopy counts may already contain matched, per-picture water correction. Do
  not subtract that blank a second time. Both OLAF count-based error bounds and
  MLE bounds change when the frozen/total counts are corrected, but neither
  separately propagates uncertainty in the measured water blank or correlations
  from sharing a blank across measurements.
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
- `src/icescopy_session_io.py`: CSV export helper and session persistence.
- `src/icescopy_plot.py`: existing PyQtGraph integration.
- `resources/preferences.xml`: bundled initial preference values.

Keep the new process runner and processing dialog in focused modules rather than
putting their calculation/control logic into the main window.

In **INP-toolkit**: `src/inptk/cli.py` defines arguments and selection,
`methods.py` validates method settings, `workflows.py` applies them, and `io.py`
defines the saved-result format. GUI integration must follow these public data and
CLI contracts rather than the private `_engine` modules.

## Verification before handoff

Validation of the final-decrease selection change is pending. Record the final
test, notebook, and installed-CLI checks here after they complete. Icescopy
integration is specified here but has not been implemented or tested.
