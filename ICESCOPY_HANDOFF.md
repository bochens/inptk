# Icescopy integration handoff

INP-toolkit owns the calculations. Icescopy edits analysis choices, displays
plots, and exchanges JSON with a separately installed `inptk` executable.
Experiments and results can stay in the toolkit process memory. Python and the CLI use the same functions and defaults. No Icescopy GUI
files have changed in this work.

## Preferences and the process dialog

Preferences → INP toolkit should contain an executable chooser, Browse, Test
connection, detected version and connection status. Test with `inptk capabilities`.
The standalone Mac installer provides `/Applications/INP-toolkit/inptk` as a stable,
user-visible executable path. Invoke it directly, without Python or a shell.
See `packaging/macos/README.md` for release signing and installer checks.
INP-toolkit **0.4.0** uses CLI protocol **2** and saved format **4**. Other formats must
be rejected explicitly. There are no compatibility flags or old-format loaders.

Preferences supply starting values. The process dialog records the choices used
for each analysis; changing Preferences must not change a saved result.

| Choice | Process dialog | Preferences |
| --- | --- | --- |
| Executable | Show connection failures | Executable path and connection test |
| `curves` | Name each output and select its physical inputs and cycle | Never store session input names globally |
| `method` | MLE or Average | Initial `mle` |
| `fit_step_C` | MLE curve shape spacing; e.g. 0.5 °C | Initial unset; hide for Average |
| `temperature_ranges_C` | Inclusive cold/warm limits for each input | Never store input-specific ranges globally |
| `temperature_step_C` | Optional grid for selecting counts before calculation | Initial off; offer 0.5 °C |
| `temperature_start_C` / `temperature_end_C` | Exact warm/cold grid endpoints, or blank to use data limits | Initial blank; do not round endpoints |
| `temperature_method` | Latest warmer, maximum warmer fraction, or centered window | Initial `latest` |
| `temperature_window_C` | Full window width, shown and required only for `window` | Optional initial width |
| `water_blank_map` | Assign raw blank sets to inputs | Never store session assignments globally |
| `water_blank_correction` | Apply blank correction checkbox, keeping assignments when off | Initial on; no map means no correction |
| `output_basis` | Suspension, sampled air or dry soil with required metadata | Avoid assuming one basis for every sample |
| `decrease_policy` | Stop at first decrease or skip decreases | Initial `stop_at_decrease` |
| `z` | Advanced uncertainty setting | Initial 1.96, nominal 95% bounds |
| `differential` | Optional intervals for individual suspension curves | Initial off |

MLE means maximum likelihood estimation: fit eligible sample and blank counts
jointly across the full cooling curve, with concentration and each run's blank
background constrained to increase or stay constant during cooling. Average takes
an equal-weight mean of eligible concentration estimates at each temperature.
Both retain blank uncertainty when raw sample and blank counts are supplied.
Average uses conservative bounds that allow shared blank uncertainty.
`--temperature-step-C 0.5` selects counts once at the beginning. Grid start/end
are optional (`--temperature-start-C`, `--temperature-end-C`); unset endpoints
use selected inputs' measured warmest/coldest values, without rounding. Both
endpoints are included, so the final interval can be shorter than the spacing.
`--fit-step-C 0.5` controls MLE curve shape independently. Concentration and
profile bounds are reported at the selected count temperatures. Plot and export
these rows directly; there is no final grid, interpolation setting, or resampled
table in the workflow.

Blank inputs are always selected explicitly by the user. Send their selected
input IDs through `--water-blank-map` for native or raw Icescopy CSV/project input.
`read_icescopy()` and CLI `--format icescopy` accept either the exported CSV or
an `.icescopy` ZIP project. Projects supply the root freeze-count CSV plus saved
physical metadata from `session.json`. No files are extracted. The reader uses
the saved exact CSV column labels to locate metadata, but never interprets their
text to choose a parent sample or blank. Missing counts and stale analysis fail
explicitly; missing physical metadata remains previewable.
Short names and long names are display text only; never infer a blank role from them.

Use every row with time, temperature and freezing counts, including rows without
an image ID. An image ID is optional source information, not an eligibility rule.
Joint MLE requires fixed total wells and cumulative first-freezing counts within
each cycle; it rejects changing totals and falling frozen counts. Raw blanks
remain separate physical well sets. Do not supply the old blank-adjusted,
changing-total export to joint MLE.
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
contributors leave a gap. There is no separate stitching or tolerance control.
An optional **Count-selection grid** enables `latest`, `max`, or `window` through
`--temperature-step-C` and `--temperature-method`. Window requires a full width
through `--temperature-window-C` (0.5 °C means ±0.25 °C). Show that field only for
window. Apply the same choice separately to sample and blank before correction.
An empty sample window stays missing; an empty required blank window is an error.
Keep this distinct from the final output grid: changing count selection refits
the analysis. Preserve and display measured temperatures alongside selected grid
temperatures in source details. CLI `capabilities` exposes flags and defaults.
Automatic Average suggestions fill these same editable controls; MLE limits
remain manually chosen.

The **Suggest Average limits** action now calls `inptk suggest-ranges`, using the
same native/saved/Icescopy input arguments and curve/cycle selection as analysis.
Its count thresholds are `--min-frozen` and `--min-unfrozen`, both initially 3.
Pass the same `--temperature-step-C`, optional start/end, method and window
settings used for analysis; suggestions now accept those same flags. Put the
thresholds in the process dialog; Preferences may supply starting values. They
are adjustable count cutoffs, not confidence thresholds. The action returns JSON and writes no
files. Applying it fills the existing range handles; manual edits stay possible.

Suggestions exhaust each dilution before switching to the next, from least to
most diluted. The first dilution keeps its initial observations, including zero
frozen wells; `min_frozen` applies only to later dilutions. `min_unfrozen` applies
to all. Later dilutions start strictly colder than the preceding stage's cold
limit. All automatic intervals are nonoverlapping; equal-dilution ties follow the
curve input order. Use the first eligible contiguous interval, without
bridging failing temperatures. This is a selection rule for Average, not MLE.
A range ends before a decrease in blank-corrected concentration. The next input
must start at or above the preceding retained concentration. Shorten the preceding
range if necessary to enable the switch; otherwise stop and report no continuation.
No output value is adjusted. These are the default automatic Average limits.
`concentration_decrease`, `below_previous_concentration`,
`shortened_for_monotone_handoff` and `no_monotone_continuation` explain these cuts.
`previous_dilution_active` explains usable points held back until a switch.
The resulting finite curve is monotone with the same analysis settings, before
any additional filter-blank spectrum subtraction; its intervals do not account
for selecting limits based on concentrations.
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

Without a count-selection grid, use original counts and temperatures. Matching observations retain their native
sequence. Alignment uses the latest observation at or warmer than the target
only where matching is needed. Native order follows observation time or original
row order. Independent streams use warm-to-cold target order. With a grid,
select counts before estimation and display the calculated temperatures directly.
There is no final regridding. MLE bounds are nominal approximate pointwise
profile intervals of the full curve likelihood, not simultaneous bands.

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
Recalculate, Cancel and Save. Use a new in-memory reference for every calculation;
write a file only when the user saves or exports.

## Persistent processing for interactive clients

Launch `inptk serve` once with stdin/stdout pipes. The process owns loaded
experiments and calculated results until released or the process closes. Send
one JSON object per line and read one response line carrying the same `id`.
No network service or temporary CSV is needed. Ordinary terminal commands still
accept file paths; `serve` also accepts those commands through `args`.

### Upload the observations once

Use the `import` request to transfer native counts and metadata directly. The
following is expanded for readability; serialize it as one JSON line:

```json
{
  "id": 1,
  "import": {
    "out": "@input-1",
    "counts": {
      "measurement_id": ["A", "A", "Water", "Water"],
      "cycle_id": ["0", "0", "0", "0"],
      "observation_id": ["frame-1", "frame-2", "frame-1", "frame-2"],
      "time_s": [0, 1, 0, 1],
      "temperature_C": [-5, -6, -5, -6],
      "n_total": [32, 32, 16, 16],
      "n_frozen": [1, 4, 0, 1]
    },
    "metadata": [
      {"measurement_id": "A", "sample_id": "Sample A", "dilution": 1, "droplet_volume_uL": 50},
      {"measurement_id": "Water", "sample_id": "Blank", "dilution": 1, "droplet_volume_uL": 50}
    ],
    "water_blank_map": {"A": ["Water"]},
    "run_id": "run-1"
  }
}
```

Both `counts` and `metadata` accept records (a list of row objects) or an object
of equal-length column arrays. Column arrays avoid repeating every field name
for every observation. These are the standard Python `read_counts` fields and
validation, not a second scientific importer. Include all valid time/temperature/
count observations, not only image rows. Carry exact identities and raw counts;
never infer blanks or groups from short names, long names, or display text.
Metadata must contain sample identity, dilution and well volume; include the
normalization metadata when requesting air or soil concentration. Missing cycle
IDs mean one cycle labelled `1`. Send explicit cycles for Icescopy.

The reply returns the reference, measurement IDs and a small table summary; it
does not echo the observations. Input validation finishes before the reference
is created. Failed imports leave existing objects untouched. Metadata or blank
assignment changes currently require a new import; there is no in-place edit
command. Method, grid, range and output-curve changes reuse the existing input.

If the input already exists as a CSV or .icescopy file, load it once with
`fractions PATH --format icescopy ... --out @input-1`. Direct JSON upload avoids
creating that file when Icescopy already holds the observations in memory.

### Calculate and retrieve plot data

Each example below is a separate request. `--format saved` means an already
imported experiment or processing result; the `@` reference stays in memory.

```json
{"id":2,"args":["suggest-ranges","@input-1","--format","saved","--temperature-step-C","0.5","--summary"]}
{"id":3,"args":["analyze","@input-1","--format","saved","--method","average","--temperature-step-C","0.5","--curves","{\"Sample A\":{\"inputs\":[\"A\"],\"cycle\":\"0\"}}","--out","@result-1"]}
{"id":4,"args":["table","@result-1","--table","cumulative","--curve","Sample A","--columns","temperature_C","concentration","lower_error","upper_error","--no-history"]}
{"id":5,"args":["save","@result-1","--out","chosen-result.inptk"]}
{"id":6,"release":["@result-1","@input-1"]}
```

Apply accepted suggestions to the next calculation with `--temperature-ranges`.
Pass the same curves and grid settings to suggestion and calculation requests.
`--summary` preserves proposed limits, reasons and completeness but omits the
large per-observation report. Only request the full report when inspecting those
observations. A projected `table` reply is plot data, not a complete saved
scientific table. Omit `--columns` and `--no-history` when the full table is needed.
Always retain units and basis from a full table or include `unit` and `basis` in
the requested columns. Error columns are distances from the concentration;
decode tagged nonfinite numbers before plotting. Display warnings from replies.

`fractions`, `estimate`, `convert`, `finalize` and `differentiate` also accept
references. `estimate` preserves concentration points before final selection;
`convert` and `finalize` reuse that estimate instead of refitting. Run
`estimate --individual` before `differentiate`. `analyze` remains the whole
workflow. Each operation uses the Python package calculations. Without `--table`,
`table @result` returns available quantities and sizes without copying their rows.
CSV export accepts `@result` directly; do not write a temporary saved result first.

### Icescopy changes required

1. Keep one toolkit process per analysis session. Discover options using
   `capabilities`; its `client_mode.import` entry advertises direct JSON upload.
2. Replace calculation-time CSV export with one import for the current source
   revision, metadata and explicit blank mapping. Retain its returned reference.
3. Send calculation settings and `--out @result-N`. Keep the last successful
   result visible while a new calculation runs; only switch after success.
4. Read just the required curves/columns using `table`. Request range summaries
   for the limits controls. Keep current unchanged-request caching in Icescopy.
5. Save or export from the retained result only on request. Release superseded
   references after success, including intermediate steps no longer needed.
6. Keep transport, large JSON parsing and plot preparation off the GUI thread.
   Reject stale replies using request IDs/source revisions. Plot temperatures
   low to high without changing the scientific row ordering in stored results.

Requests run sequentially; coalesce obsolete pending edits instead of queueing
all slider positions. References cannot be overwritten: calculate a fresh name,
then release the old one. Released inputs can remain alive through retained
results that reference their experiment; release those results too to reclaim
memory. There is no hidden growing cache across requests. Closing stdin exits;
terminating the process cancels work and discards all unsaved references. Restart
and reimport after a process crash. Preserve the last displayed successful result.

The current Icescopy implementation still uses file paths for calculations; these
client changes are not applied by an INP-toolkit update alone. Saved format 4 and
the default full response fields are retained; saved JSON now omits whitespace.
The installed editable package uses current source, but a running process keeps
already imported code. Restart it after package updates.

`examples/benchmark_client.py` measures upload, repeated requests and response
sizes through a real subprocess without writing data files.
`examples/benchmark_processing.py` measures calculations without transport.

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

Saved `analysis.json` format 4 contains experiment, frozen_fraction, curves,
settings, history and warnings. Each curves[name] contains curve_id, sources and
its tables: cumulative, excluded, and optional differential. Tables
contain type, columns, dtypes, rows and history. There are no stage-based result
fields. The whole original experiment, including unused observations, is retained.

Curve rows have sample_id, curve_id, point_id and point_order. Cross-run curves
do not invent a single physical run/cycle identity. Contributor lists and
source_observations record actual sample/blank identities, temperatures and
alignment. lower_error and upper_error are **widths**, not interval endpoints:
subtract/add them to concentration. Preserve unit, basis and quality flags.
Nonfinite numbers use objects such as {"$nonfinite":"inf"}.

CSV export defaults to cumulative. Select --table excluded
explicitly. Omitting --curve exports all curves, retaining curve_id labels.
The CSU helper also selects --curve and formats saved sampled-air concentrations;
it performs no separate calculation or normalization.

Failures contain error.code and error.message. Exit codes: 0 success, 1 invalid
input/processing, 2 argument error, 130 interrupted. A crashed process may not
produce complete JSON; retain the last successful result. Duplicate JSON names
are rejected. JSON choices can be passed directly or through files, each as a
separate argument. Never overwrite previous outputs.
