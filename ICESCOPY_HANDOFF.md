# Icescopy integration handoff

INP-toolkit owns the calculations. Icescopy edits analysis choices, displays
plots, launches a separately installed `inptk` executable and reads its saved
results. Python and the CLI use the same functions and defaults. No Icescopy GUI
files have changed in this work.

## Preferences and the process dialog

Preferences → INP toolkit should contain an executable chooser, Browse, Test
connection, detected version and connection status. Test with `inptk capabilities`.
INP-toolkit **0.4.0** uses CLI protocol **2** and saved format **3**. Other formats must
be rejected explicitly. There are no compatibility flags or old-format loaders.

Preferences supply starting values. The process dialog records the choices used
for each analysis; changing Preferences must not change a saved result.

| Choice | Process dialog | Preferences |
| --- | --- | --- |
| Executable | Show connection failures | Executable path and connection test |
| `curves` | Name each output and select its physical inputs and cycle | Never store session input names globally |
| `method` | MLE or Average | Initial `mle` |
| `temperature_ranges_C` | Inclusive cold/warm limits for each input | Never store input-specific ranges globally |
| `water_blank_map` | Assign raw blank sets to inputs | Never store session assignments globally |
| `water_blank_correction` | Apply blank correction checkbox, keeping assignments when off | Initial on; no map means no correction |
| `output_basis` | Suspension, sampled air or dry soil with required metadata | Avoid assuming one basis for every sample |
| `decrease_policy` | Stop at first decrease or skip decreases | Initial `stop_at_decrease` |
| `output_step_C` | Optional final grid spacing | Initial off |
| `output_method` | Sample or interpolate the final curve | Initial `sample` |
| `z` | Advanced uncertainty setting | Initial 1.96, nominal 95% bounds |
| `differential` | Optional intervals for individual suspension curves | Initial off |

MLE means maximum likelihood estimation: fit eligible sample and blank counts
jointly across the full cooling curve, with concentration and each run's blank
background constrained to increase or stay constant during cooling. Average takes
an equal-weight mean of eligible concentration estimates at each temperature.
Both retain blank uncertainty when raw sample and blank counts are supplied.
Average uses conservative bounds that allow shared blank uncertainty.

Export raw counts for actual images: fixed total wells and cumulative first-freezing
counts within each cycle. Do not supply the old blank-adjusted, changing-total
export or temperature-clock rows filled between images. Joint MLE rejects changing
totals and falling frozen counts. Raw blanks remain separate physical well sets.
The CLI `capabilities` response describes these method requirements.

Grouping, runs, cycles, dilution, actual droplet volume, sample normalization and
blank assignments belong to the analysis session. Do not infer them from similar
names. Missing physical metadata prevents concentration calculation; counts and
fractions can still be previewed. Temperature import should attach temperatures
and times. The INP process dialog owns blank selection and grouping. Connect that
replacement before removing the old import-stage blank picker.

## Named curves

A name identifies an output, not another physical sample or another observation.
Use exactly the same specification in Python and in the CLI `--curves` JSON:

```json
{
  "Sample A": {"inputs": ["Sample_0", "Sample_1"], "cycle": "1"},
  "Sample A neat": {"inputs": ["Sample_0"], "cycle": "1"},
  "Sample A diluted": {"inputs": ["Sample_1"], "cycle": "1"}
}
```

One input produces an individual curve; several independent inputs produce a
combined curve, with equal or different dilution factors. Keeping individual and
combined outputs does not increase the number of independent droplets. Names
must be unique, nonempty text; preserve the exact name through plots and export.

Omit `cycle` only if every named input has exactly one observed cycle. Multiple
cycles require explicit selection. Omitting `curves` generates one output per
original sample/run/cycle. Repeated freezing cycles never become independent
replicates and are never pooled.

To combine independent inputs from runs with different cycle labels, use:

```json
{
  "Sample A across runs": {
    "inputs": [
      {"measurement_id": "Sample_0_run1", "cycle_id": "01"},
      {"measurement_id": "Sample_0_run2", "cycle_id": "02"}
    ]
  }
}
```

Metadata supplies run and original-sample identity. Each curve requires one
original sample and at most one cycle per run. Duplicate inputs, unknown inputs,
blank inputs, mixed parent samples and mixed cycles in one run are errors. Each
run keeps its own assay-blank background. Explicit choices calculate only the
requested curves; unrequested blank coverage cannot block the calculation.

## Plots and scientific choices

Use plots as the main view, with tables available for inspection and export.
Show original counts, observed frozen fractions and named concentration curves.
Offer individual curves as overlays, using the same curve selector as combined
curves. Give each input a stable color and a readable name/dilution label.

Temperature increases **left to right**, for example −30°C → −5°C. Display order
must not change saved observation order. Show holds and temperature reversals;
do not merge original count rows just because they have the same temperature.
The MLE fitted curve has one value per distinct native temperature, ordered warm
to cold. Show it separately from the original frozen-fraction observations.

Provide inclusive cold (`min_C`) and warm (`max_C`) handles and numeric fields for
each input. Omitted boundaries are unlimited. Ranges apply to targets and source
observations, shared across selected cycles. MLE fits all eligible histories
together; Average combines eligible estimates at each temperature. Zero
contributors leave a gap. There is no
separate stitching control, temperature-selection mode or tolerance control.
Automatic Average suggestions fill these same editable controls; MLE limits
remain manually chosen.

The **Suggest Average limits** action now calls `inptk suggest-ranges`, using the
same native/saved/Icescopy input arguments and curve/cycle selection as analysis.
Its two settings are `--min-frozen` and `--min-unfrozen`, both initially 3. Put these
in the process dialog; Preferences may supply starting values. They are adjustable
count cutoffs, not confidence thresholds. The action returns JSON and writes no
files. Applying it fills the existing range handles; manual edits stay possible.

Suggestions exhaust each dilution before switching to the next, from least to
most diluted. The first dilution keeps its initial observations, including zero
frozen wells; `min_frozen` applies only to later dilutions. `min_unfrozen` applies
to all. Later dilutions start strictly colder than the preceding stage's cold
limit. Equal-dilution inputs may overlap and be averaged; switch only after all
inputs at that dilution end. Use the first eligible contiguous interval, without
bridging failing temperatures. This is a selection rule for Average, not MLE.
`previous_dilution_active` explains usable points held back until that switch.
Requests sharing an input across different curve input sets must be made
separately, because their switching limits can differ. Manual overlap is allowed.

Use `inputs` to show each proposed interval, cycle, cutoff reasons and counts of
retained/excluded observations. Use `table` to overlay eligibility and blank flags
on the original observations. `not_distinguished_from_blank` means the individual
corrected concentration interval reaches zero; display it without dropping that
point. `uncertainty_unavailable` needs attention but is not another hidden cutoff.
Missing blank coverage is an eligibility failure when correction is enabled.

If `complete=false`, the combined `temperature_ranges_C` field is null. Do not
pass it to analysis, where null means unrestricted: require the user to resolve
inputs with `status="no_usable_range"`, adjust thresholds, or change the selected
inputs and request suggestions again. The valid individual proposals remain
visible. One cycle per measurement is required; request other cycles separately.
Keep the suggestion report in the session alongside any subsequent manual edits;
the saved analysis records the applied ranges, not the discarded suggestions.
Suggestions use count thresholds, not corrected-concentration peaks. They cannot
guarantee a monotone Average curve when contributors change.

Use original counts and temperatures. Matching observations retain their native
sequence. Alignment uses the latest observation at or warmer than the target
only where matching is needed. Native order follows observation time or original
row order. Independent streams use warm-to-cold target order. Sampling or
interpolation happens afterward and is a separate view.
For MLE, the complete curve is fitted before any sampling; changing the display
grid cannot change the fit or its uncertainty. Bounds are nominal approximate
pointwise profile intervals of the full curve likelihood, not simultaneous bands.

Show retained `cumulative` points normally and `excluded` points faintly with
reasons. `stop_at_decrease` excludes the first decrease and all later points;
`skip_decreases` excludes dips and permits recovery. Neither creates a plateau
or changes uncertainty. A fixed relative margin of 1e-9 only allows numerical
roundoff. Respect `segment_id` when joining points or drawing uncertainty bands.
Resampling never bridges exclusions or adds fitted statistical information.
The MLE curve is already monotone, so these decrease choices normally affect only
Average or a curve after additional sample/filter-blank subtraction.

Raw blanks may have different well totals and droplet volumes. The model assumes
all matched-assay background scales with liquid volume, including water,
substrate and well-wall contributions. Eligible points lacking blank coverage
raise an error. Previously corrected counts cannot recover missing raw blank
uncertainty. The Python-only `blank_by_curve` is a separate subtraction of an
already calculated sample/filter blank spectrum and is not a CLI control.

Differential output currently requires one input per curve, suspension basis,
and no additional sample/filter blank spectrum. It retains only intervals whose
original endpoints survived selection. Combined differential output is not
implemented; do not label individual-input intervals as combined intervals.

Keep edited controls separate from the last calculated result. Provide
Recalculate, Cancel and Save. Use a new output path for every calculation.

## CLI and saved-result contract

Use Qt separate-process support with an executable path and argument list,
not a shell command. Capture stdout, stderr and exit status. A successful JSON
reply is one object, even when `--json` appears before the subcommand.

```sh
inptk capabilities
inptk preview raw-counts.csv --metadata measurements.csv --json
inptk analyze raw-counts.csv --metadata measurements.csv \
  --curves curves.json --water-blank-map blanks.json \
  --temperature-ranges ranges.json --output-basis sampled_air \
  --out analysis.inptk --json
inptk export-csv analysis.inptk --curve 'Sample A' --out sample-a.csv --json
```

Native counts use measurement_id, cycle_id, temperature_C, n_total and n_frozen.
Optional observation_id, picture_id and time_s preserve acquisition identity.
Measurement metadata records measurement_id, sample_id, run_id, dilution and
**droplet_volume_uL**, plus parent-sample normalization fields. Every Icescopy
export must retain the actual volume and raw sample/blank counts. `water_blank_map`
assigns each sample input a list of same-run blank input names. Shared blanks
are counted once in joint fitting. Blanks have dilution 1; cycles stay separate.

`preview` accepts incomplete metadata and returns original counts/fractions,
measurement summaries, provisional sample assignments and missing physical
fields. It performs no fitting or correction and writes no files. Its metadata
check does not establish blank coverage or air/soil normalization validity.

`capabilities` reports actual parser flags, defaults, choices, observation tables
and curve quantities. All replies include protocol_version, toolkit_version,
saved_format_version, command, status and warnings. Analyze replies include
output, reusable settings, observation_tables and a curves dictionary. Each
curve entry contains curve_id, kind, sources and table types/row counts.

Saved `analysis.json` format 3 contains experiment, frozen_fraction, curves,
settings, history and warnings. Each curves[name] contains curve_id, sources and
its tables: cumulative, excluded, and optional resampled/differential. Tables
contain type, columns, dtypes, rows and history. There are no stage-based result
fields. The whole original experiment, including unused observations, is retained.

Curve rows have sample_id, curve_id, point_id and point_order. Cross-run curves
do not invent a single physical run/cycle identity. Contributor lists and
source_observations record actual sample/blank identities, temperatures and
alignment. lower_error and upper_error are **widths**, not interval endpoints:
subtract/add them to concentration. Preserve unit, basis and quality flags.
Nonfinite numbers use objects such as {"$nonfinite":"inf"}.

CSV export defaults to cumulative. Select --table resampled or --table excluded
explicitly. Omitting --curve exports all curves, retaining curve_id labels.
The CSU helper also selects --curve and formats saved sampled-air concentrations;
it performs no separate calculation or normalization.

Failures contain error.code and error.message. Exit codes: 0 success, 1 invalid
input/processing, 2 argument error, 130 interrupted. A crashed process may not
produce complete JSON; retain the last successful result. Duplicate JSON names
are rejected. JSON choices can be passed directly or through files, each as a
separate argument. Never overwrite previous outputs.
