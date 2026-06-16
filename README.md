
<img src="assets/ufolaf-logo.png" align="right" height="90" />

# UFOLAF
UFOLAF is an unofficial fork of OLAF (OpenSource Library for Automating Freezing data acquisition from Ice Nucleation Spectrometer). It is maintained independently and is not affiliated with or endorsed by the original OLAF authors.

UFOLAF was created to support easier use as an external tool with Icescopy, another image-analysis software package for ice nucleation freezing-array experiments. UFOLAF remains licensed under the GNU Affero General Public License v3.0 and is intended to operate separately from Icescopy so as not to affect Icescopy's MIT licensing.

This project is based on OLAF. Original OLAF documentation is available [here](https://sigran.github.io/OLAF/). If you use UFOLAF in research, please also cite the original OLAF release: https://doi.org/10.5281/zenodo.17509699

UFOLAF preserves the original OLAF copyright and license notices. Modifications in UFOLAF are identified as changes from the original project.

## Install for Development

```bash
python -m pip install -e ".[dev]"
```

After installation, both import styles work:

```python
import ufolaf

counts = ufolaf.read_counts("freeze_count_timeseries.csv")
```

```bash
ufolaf --help
python -m ufolaf --help
```

## CLI Artifact Format

The CLI writes UFOLAF artifact directories rather than plain CSV by default.
Each artifact keeps table data, sample metadata, and processing metadata
together:

```text
result.ufolaf/
  manifest.json
  data.csv
  sample_metadata.json
  processing_metadata.json
```

List and dictionary outputs are stored as nested artifact directories. Use
`ufolaf export-csv` when only the table data are needed.
