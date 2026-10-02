# Icescopy integration handoff

INP-toolkit performs the calculations. Icescopy supplies controls and plots,
launches the separately installed `inptk` executable, and reads its saved results.
Do not copy the calculations into Icescopy or import the Python package there.
No Icescopy GUI files have changed in this work.

## Preferences and analysis settings

Add **Preferences → INP toolkit** with an executable chooser, **Browse**, **Test
connection**, detected version, and connection status. Test with `inptk capabilities`
and check the returned protocol and saved-format versions.
Use Icescopy's existing preference location and atomic save mechanism.

Preferences provide starting values. The process dialog shows and saves the
values actually used; changing Preferences must not change a saved analysis.

| Setting | Process dialog | Preferences |
| --- | --- | --- |
| `method` | MLE or Average, using the same eligible observations | Default `mle` |
| `combination_groups` | Choose exact measurement/cycle members for each named output group | Never store session identities globally |
| `temperature_ranges_C` | Inclusive cold/warm limits for each measurement | Never store measurement-specific limits globally |
| `water_blank_map` | Assign raw blank sets to sample measurements | Never store measurement names globally |
| `water_blank_correction` | Visible Apply blank correction checkbox; retain assignments when off | Initial value on; no map means no correction |
| `output_basis` | Suspension, sampled air, or dry soil, with required metadata | Do not assume one basis for every sample |
| `decrease_policy` | Stop at first decrease, or skip decreases and allow recovery | Default `stop_at_decrease` |
| `output_step_C` | Optional final-result grid spacing; off preserves native output only | Default off; optional preferred spacing |
| `output_method` | Sample existing final values or interpolate final values | Default `sample`; relevant only with a grid |
| `z` | Advanced uncertainty setting | Default 1.96, for nominal 95% bounds |
| `differential` | Additional per-measurement concentration-per-degree output | Default off |

MLE, maximum likelihood estimation, fits the eligible droplet counts together.
Average takes the equal-weight mean of their concentration estimates. Average
uses conservative bounds that allow shared blank uncertainty; MLE uses counts
jointly. Keep this explanation beside the method choice.

There are no temperature-selection-method or tolerance controls. Calculations
start from original counts and temperatures. Latest-warmer alignment is automatic
only when observations need matching. Output spacing changes the optional final
view, not the input observations used for estimation.

Sample grouping, measurement/run/cycle names, dilution factors, droplet volumes,
air/suspension/soil amounts, and blank assignments are analysis data. Keep them in
the session. Never infer sample relationships or volumes from similarly named
measurements. The Python-only `blank_by_group` subtracts an already calculated
sample/filter blank curve; it is distinct from raw assay-blank assignment and
has no CLI control.

## Process dialog and plots

Temperature import should attach temperatures and times only. The new INP dialog
owns sample grouping, metadata review, raw blank assignment, measurement ranges,
and final-curve selection. Remove the old import-stage blank picker once this
replacement is connected; existing saved assignments may prefill the controls.
Preserve raw observations before correction.

- Review measurement identities, dilution and actual droplet volumes. Group
  measurements from the same original sample.
- By default, keep each sample/run/cycle separate. Offer explicit named groups
  for independent measurements from different runs. Each group must select only
  one cycle per run; repeated freezing cannot create extra independent droplets.
- Make blank correction optional and show blank counts and curves. State the
  assumption that all measured blank sources, including water, substrate and
  PCR well walls, scale with liquid volume. Sample and blank volumes may differ.
- Use plots as the main view: counts, frozen fraction, per-measurement
  concentration, combined concentration, final selection, and optional grid or
  differential results. Keep tables available for inspection and export.
- Temperature increases **left to right**, for example **−30°C → −5°C**. Display
  order must not change saved observation order or calculations. Show points and
  source information for holds or temperature reversals; do not silently merge
  rows sharing a temperature.
- Give each measurement a stable color and a name/dilution label. Provide cold
  (`min_C`) and warm (`max_C`) handles, numeric entries, and a no-limit state.
  Shade excluded temperatures while retaining the full frozen-fraction curve.
- Overlapping ranges use every eligible measurement through MLE or Average.
  Disjoint ranges can select one dilution at a time. One contributor uses its
  estimate and uncertainty; zero contributors produce a gap. There is no
  separate stitching control or automatic minimum-unfrozen cutoff.
- Show contributors and their actual observed temperatures for each combined
  point. Any future automatic range suggestion should populate the same explicit
  controls for review.
- Draw excluded `final_candidates` faintly with their reasons. Overlay retained
  `final` points; never invent a plateau. Keep `resampled`, when requested, a
  separately labelled view. Respect `segment_id` when joining points.
- Separate edited controls from the last calculated result. Provide
  **Recalculate**, **Cancel**, and **Save**, and show returned errors and warnings.

Ranges are inclusive and apply to both a calculation target and the original
row selected to supply it. Each measurement has one continuous range, shared
across selected cycles; separate jobs are needed for cycle-specific limits.
Omitted measurements use their full support. Exclusions do not remove frozen
counts from the remaining observations or require blank coverage at excluded
points. Eligible observations with missing blank coverage raise an error.

Native sequences retain time order, with stable ties, or original input order
when time is absent. They keep repeated temperatures, holds, reversals, and later
warming. Final selection follows that observation order: `stop_at_decrease`
excludes the first lower concentration and all later points; `skip_decreases`
excludes a dip but allows later recovery to the last retained value. Aligned
cross-run groups use their explicit warm-to-cold target order.

Numerical equality allows a fixed relative margin of `1e-9` and zero absolute
margin, recorded in history. It only prevents solver roundoff from triggering a
decrease. Nonfinite values do not become comparison values; negative finite
values are not clipped. Neither choice changes concentrations or error bounds.

## CLI input and process contract

Use Qt's separate-process support with an executable path and argument list,
not a shell command. Read results only after successful completion. Capture
standard output, standard error, and exit status: 0 success, 1 input/processing
failure, 2 argument-parsing error, 130 interrupt. Use `--json` and parse one complete
JSON response from standard output. Keep standard error for diagnostics.
Every output path must be new. Preserve prior successful results after failed
jobs and remove disposable previews when no longer needed.

The CLI provides a versioned interface for interactive use:

1. Run `inptk capabilities`. Read `protocol_version`, `toolkit_version`,
   `saved_format_version` and `commands`. Each command lists its actual flags,
   defaults, choices and repeatable arguments. Protocol 1 and saved format 2 are
   implemented here; do not scrape help text or silently accept unknown versions.
2. Run `inptk preview raw-counts.csv --json`. Add `--metadata measurements.csv`
   when available, even if physical fields are incomplete. `--format icescopy`
   reads old exports; `--format saved` previews the original saved experiment.
   The returned `table` contains frozen fractions and their original counts,
   temperatures and identities, ready for plots. Preview writes no files and
   performs no fitting, blank correction, alignment or aggregation.
3. Use `measurement_metadata`, `measurements`, `provisional_sample_assignments`
   and `suspension_metadata.missing_fields` to populate the process dialog.
   Unassigned measurements remain separate. `suspension_metadata.valid` checks
   metadata only: selected blank coverage, groups, ranges and air/soil conversion
   still need validation during calculation.
4. Run `analyze ... --out new-preview.inptk --json` after the user chooses settings.
   A successful reply contains `output` (absolute path), resolved `settings`,
   table row counts and warnings. Read the saved tables only after exit code 0
   and `status == "ok"`. Retain the last successful plot if recalculation fails.

Except for human `--help` and `--version`, `--json` produces one response object
with `protocol_version: 1`, `toolkit_version`, `saved_format_version`, `command`,
`status` and `warnings`. Errors add `error.code` and `error.message`. Codes include
`usage_error`, `invalid_input`, `output_exists`, `file_not_found`,
`permission_denied`, `io_error`, `cancelled` and `internal_error`. Unexpected
failures also put a traceback on standard error. Signals or a crashed executable
can prevent a response; treat missing/truncated JSON as failure, not an empty
successful result. Nonfinite-number encoding matches the saved-file contract.

Icescopy uses PySide6. Use `QProcess.start(executable, argument_list)` and its
`finished`/`errorOccurred` signals; do not call `waitForFinished()` on the GUI
thread. Read stdout and stderr separately. Associate each process with the
settings that launched it, discard stale replies, and cancel its process when a
request is superseded. Cancellation must leave previously saved results intact.
Use a temporary directory per preview job and a new result subdirectory; save the
selected successful result only when the user chooses Save.

Export **raw sample and blank counts** to native long CSV for new analyses. The
existing `freeze_count_timeseries.csv` may already be corrected and does not
provide complete raw blank context; never reverse a correction to reconstruct
observations. Native count columns are:

```csv
measurement_id,cycle_id,observation_id,time_s,temperature_C,n_total,n_frozen
Sample_0,1,s0-0001,0,-5.01,32,0
Sample_1,1,s1-0001,0,-5.01,32,0
Water_1,1,b1-0001,0,-5.01,20,0
```

Supply `time_s` and stable `observation_id` when available. Missing observation
IDs are generated once on import. `picture_id` is optional image identity and
is separate from the row ID. Keep instrument observations without pictures.
Repeated temperatures are valid; zero-total rows are rejected and any source
exclusion must be explicit. Each physical droplet set has a unique measurement
ID, including across runs, and one metadata row:

```csv
measurement_id,sample_id,run_id,dilution,droplet_volume_uL,sample_type,air_volume_L,suspension_volume_mL,filter_fraction_used
Sample_0,Sample_A,run-01,1,50,air,100,5,1
Sample_1,Sample_A,run-01,10,50,air,100,5,1
Water_1,water1,run-01,1,20,other,,,
```

These values are examples; use recorded metadata. Native `sample_id` groups
measurements without `--sample-map`. Every set needs a known positive volume;
blank sets have dilution 1 and the same run as their assigned sample sets.

`water-blank-map.json` maps each nonblank measurement to a list, even for one blank:

```json
{"Sample_0": ["Water_1"], "Sample_1": ["Water_1"]}
```

Dilutions of one original sample in one run must use the same blank sets. Each
blank's rows appear once, even when many samples use it. Matching cycles are
required; repeated blank cycles are never pooled. Different runs retain their
own blank observations and separately fitted backgrounds.

`temperature-ranges.json`:

```json
{"Sample_0": {"min_C": -15}, "Sample_1": {"max_C": -12}}
```

Sample_0 contributes at −15°C and warmer; Sample_1 at −12°C and colder. Both
contribute between −15°C and −12°C. Omit a bound or set it to `null` for no limit.

Optional `combination-groups.json` explicitly requests output groups:

```json
{
  "Sample_A_replicates": [
    {"measurement_id": "Sample_0", "cycle_id": "1"},
    {"measurement_id": "Sample_1", "cycle_id": "1"},
    {"measurement_id": "Sample_run2", "cycle_id": "2"}
  ]
}
```

Here Sample_run2 would require its own count rows, metadata and, when correction
is enabled, same-run blank assignment. Members must have one parent sample and
at most one cycle per run. Run identity is read from measurement metadata, not
supplied in the group JSON. Duplicate, unknown or blank members are errors.
Without this file, groups remain separate by sample/run/cycle. An explicit mapping
requests only the listed groups. The full workflow also limits `per_dilution`
and optional `differential` estimates to those measurement/cycle members, keeping
their required blank context. Missing blank coverage in an unrequested run cannot
block the requested fits. The saved `Experiment` and full `frozen_fraction` table
remain unchanged. Default analysis without explicit groups is unchanged.

```sh
inptk analyze raw-counts.csv --format native --metadata measurements.csv \
  --water-blank-map water-blank-map.json --sample Sample_A \
  --method mle --temperature-ranges temperature-ranges.json \
  --output-basis sampled_air --decrease-policy stop_at_decrease \
  --out analysis.inptk
```

Add `--combination-groups combination-groups.json` for explicit groups. Both this
argument, `--temperature-ranges`, `--sample-map` and `--water-blank-map` accept
a JSON object directly or a file path.
For an optional final grid add `--output-step-C 0.5 --output-method sample` or
`interpolate`. There are no pre-fit `--step-C`, `--temperature-method`,
`--temperature-tolerance-C`, `--dilution-method`, or `--method-options` arguments.

`--sample` and `--cycle` select exact parent sample/cycle IDs and are repeatable.
Required blank rows and metadata are retained once; blank-only parent samples
cannot be analysis selections. `source.selection` records requested selections.
`--run-id` labels imports, not a saved-run filter. Use group membership to choose
particular measurements and runs.

`--no-water-blank-correction` retains raw blank context but analyzes sample counts
without correction; blank coverage is then unnecessary. Show the returned
`water_blank_correction_applied`: a checked box without a map does not imply a
correction occurred. Neither state reverses correction in imported counts.

`--format saved` reuses the original saved experiment and rejects replacement
metadata/maps. It does **not** reuse prior processing settings; supply the chosen
groups, ranges and other options again. The older `--format icescopy` adapter
accepts `--sample-map` but rejects `--water-blank-map`. Its overrides use exported
measurement names as `sample_id` and `well_volume_uL` for volume, such as
`[{"sample_id":"Sample_0","well_volume_uL":50}]`. Native metadata instead uses
`measurement_id` and `droplet_volume_uL`.

## Alignment and saved results

One sample keeps every native observation. Synchronized same-run sample streams
keep their shared native states. Matching time/temperature sequences establish
this correspondence; without time, identical unique-temperature sequences also
suffice. Generated row numbers alone do not prove synchronization.

Only unmatched streams or cross-run groups need alignment. Targets are the union
of original sample temperatures, unique and warm-to-cold. Select the latest
eligible source at or warmer than each target, with zero allowance for colder
values. Pairwise matching sample/blank acquisitions move together even when another
sample has missing rows. If members request different acquisitions of a shared
blank, select its latest-warmer state at the target once and record alignment.
Otherwise blanks use latest-warmer matching. Counts are never interpolated; observations outside their
physical temperature support are never extrapolated. No first-minimum trim or
synthetic zero counts are introduced.

`analysis.inptk` is a directory containing `analysis.json`. Check
`format == "inptk"` and `format_version == 2`; other versions are rejected.
The payload includes `toolkit_version`, original counts, metadata, raw blank map,
settings, tables, history and warnings. Settings include the normalized groups
(with resolved sample/run identities), ranges, method, optional output grid, and
requested/effective water-correction states. `water_blank_model="volume_scaled"`
records the fixed physical assumption, not another user control.

Tables contain columns and row records. Nonfinite numbers use explicit objects
such as `{"$nonfinite":"inf"}`. Decode them and preserve quality flags.
`lower_error` and `upper_error` are widths: subtract/add them to concentration
for interval endpoints.

- `frozen_fraction`: original observed counts and fractions, with physical
  measurement/run/cycle/observation identities.
- `per_dilution`: native per-measurement concentration, including rows outside
  ranges with missing estimates and `selection_status="outside_temperature_range"`.
- `combined`: `sample_id`, `group_id`, `point_id`, and `point_order`; it does not
  fabricate a single run/cycle label for cross-run results. `source_observations`
  is a JSON list of actual sample/blank row identities, observed temperatures,
  optional times, and alignment labels. Contributor lists and `contributor_count`
  explain each point; `selection_status` is `single`, `combined`, or
  `no_eligible_measurements`. Nonfinite fitted values carry separate quality flags.
- `final_candidates`: all post-correction/post-conversion rows, `used_in_final`,
  and `final_selection_status`: `kept`, `nonfinite`, `decrease`, or `after_decrease`.
- `final`: retained original calculation points, with `segment_id` preserving gaps.
- Optional `resampled`: a separate final-grid view. `sample` copies the latest
  warmer retained value and errors. `interpolate` first applies the same rule
  at each observed temperature to form ordered-temperature endpoints, then
  connects their concentration and interval endpoints linearly. This avoids
  reintroducing decreases by sorting native temperature reversals. These are not
  newly fitted intervals. Actual source point IDs and temperatures remain in
  `source_point_ids` and `source_temperatures_C`; `endpoint_temperatures_C`
  separately records the interpolation endpoint temperatures. Sampling and
  interpolation flags are recorded. Neither method adds extrapolation or bridges
  excluded segments; existing source extrapolation flags propagate. Original
  `final` values and uncertainties remain unchanged.
- Optional `differential`: per-measurement changes between adjacent strictly
  cooling observations divided by the actual temperature interval. Holds/warming
  have no cooling interval and are recorded in history, without bridging them.

Save the input identities, executable/version, requested options and returned
settings with the Icescopy analysis. Concentration requires dilution and droplet
volume; air/soil output requires its additional metadata. The preview command
below needs neither dilution nor volume and does not estimate concentrations.

```sh
inptk export-csv analysis.inptk --table final --out final.csv
inptk export-csv analysis.inptk --table resampled --out grid.csv
```

Request `resampled` only if present. For the optional CSU layout, format one
already calculated sampled-air group:

```sh
python scripts/csu_inp_processing.py analysis.inptk --out csu.csv \
  --group Sample_A_replicates --allow-missing-header
```

The helper also supports `--table resampled`. It never refits, normalizes or
reselects points. `--header KEY=VALUE` supplies header fields; recorded
normalization metadata cannot conflict. The CSV columns are `degC`, `dilution`,
`INPS_L`, `lower_CI`, and `upper_CI`, with CI columns containing error widths.
Mixed-dilution points have blank dilution. Outputs must be new paths outside the
saved analysis folder.

## Scientific boundaries

- All measured blank sources are represented as a background per unit liquid
  volume, with a separate background for each run/selected cycle. Actual sample
  and blank volumes enter the model; equal volumes are not required. There is no
  surface-area term or background-model selector.
- MLE fits a common original-sample concentration and each run's own background.
  It cannot identify contamination unique to one dilution and absent from its
  assigned blanks. Measurement temperature ranges remain an explicit decision.
- A blank with 32 wells is not subtracted as 32 droplets from a sample with four
  wells. At the same frozen fraction and volume, more wells imply the same
  background level with greater precision. Each physical blank enters a joint
  point fit once, regardless of how many dilutions share it.
- MLE profile bounds allow fitted backgrounds to vary while finding concentration
  limits. The threshold is `z**2/2`, default `z=1.96`. Average widens each
  measurement's marginal profile interval for the number of contributors, then
  averages lower and upper endpoints. These **Bonferroni-adjusted marginal profile
  bounds** allow dependence from shared blanks. The
  [Bonferroni adjustment](https://www.itl.nist.gov/div898/handbook/prc/section4/prc463.htm)
  does not require independence, but the profile intervals remain approximate.
  Do not promise exact 95% coverage or smaller bounds with more measurements.
- One contributor keeps its ordinary estimate and bounds. An infinite individual
  estimate makes the average infinite; an unavailable estimate makes it
  unavailable, without silently omitting contributors.
- Previously corrected counts cannot recover raw blank uncertainty. Their bounds
  are conditional on those counts; do not correct them again or reverse clipped
  subtraction to claim raw observations.
- Temperatures and repeated cycles do not add independent physical droplets.
  Supplied volume/dilution errors and cycle variability are not included in the
  count intervals. Unexplained droplet loss needs separate scientific interpretation.

## Implementation locations

In the **Icescopy repository**:

- `src/icescopy_aux.py`: `PreferencesDialog`, categories, and existing Browse layout.
- `src/icescopy_paths.py`: preference location and atomic saving.
- `src/Icescopy.py`: apply preferences and register the analysis action.
- `src/icescopy_dialogs.py`: existing temperature-import blank controls and output
  dialog; move blank assignment to the new INP process dialog.
- `src/icescopy_freeze_count_timeseries.py` and `src/icescopy_temperature_import.py`:
  raw count preparation and existing blank subtraction; preserve counts before it.
- `src/icescopy_session_io.py`: session persistence/current export; add raw native
  export without replacing existing saved results.
- `src/icescopy_plot.py`: PyQtGraph integration.
- `resources/preferences.xml`: initial preferences.

Keep the process runner and dialog in focused modules. In **INP-toolkit**,
`cli.py` handles arguments and selections; `processing.py` supplies native steps;
`alignment.py` matches observed states; `workflows.py` combines groups and selects
final points; `resampling.py` creates optional final grids; `water_blank.py`
prepares background fits; `methods.py` validates groups and ranges; `io.py` defines
the saved format. Integrate with the public CLI/data contract, not `_engine`.

## Verification before integration

- All 603 tests passed. The deliberate invalid-timestamp case emits one pandas
  parsing warning. Ruff, mypy across 21 source files, Bandit and diff checks passed.
- Wheel and source distributions built. The installed wheel was checked outside
  the checkout with MLE/Average, raw blanks on/off, unequal volumes, exact cycle
  selection, explicit cross-run groups, saved-format-2 replay, native/resampled
  CSV export and refusal to overwrite existing results.
- The M1 notebook executed 13 code cells and generated five figures. Separate
  calls matched the complete workflow; the input CSV checksum was unchanged.
  With its configured ranges, the first decrease occurs at −10°C and the default
  policy retains 5,278 of 8,705 native calculation points. All candidates remain
  available. This legacy corrected-count file cannot recover raw blank uncertainty.
- The installed executable completed `capabilities`, metadata-free `preview`
  and `analyze` through PySide6 6.9.3 `QProcess` on 2,400 synthetic observations.
  A timer continued firing during each command, verifying that the caller's Qt
  event loop remained responsive. Protocol tests also cover structured errors,
  interrupts, unexpected failures, inline mappings and preservation of input files.
- The synthetic standard-workflow example ran successfully. The notebook keeps
  external export disabled; existing source data and result folders were preserved.

The build emits a nonblocking deprecation warning for the existing license
metadata table in `pyproject.toml`; this work does not change the licence terms.
No Icescopy GUI integration or GUI testing has been performed in this work.
