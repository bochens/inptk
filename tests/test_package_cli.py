from __future__ import annotations

import subprocess
import sys

import numpy as np

import ufolaf


def test_public_package_imports() -> None:
    assert ufolaf.__version__ == "0.1.0"
    assert ufolaf.CountsTable.__name__ == "CountsTable"
    assert callable(ufolaf.read_artifact)
    assert callable(ufolaf.write_artifact)


def test_artifact_roundtrip(tmp_path) -> None:
    metadata = ufolaf.SampleMetadata(
        sample_id="sample-1",
        well_volume_uL=50,
        dilution=1,
    )
    table = ufolaf.CountsTable(
        sample_id=["sample-1"],
        temperature_C=[-5.0],
        n_total=[10],
        n_frozen=[3],
        metadata=metadata,
        processing_metadata=ufolaf.processing_metadata_for(
            "test_artifact_roundtrip",
            source_sample_ids=("sample-1",),
        ),
    )

    path = tmp_path / "counts.ufolaf"
    ufolaf.write_artifact(table, path)
    restored = ufolaf.read_artifact(path)

    assert isinstance(restored, ufolaf.CountsTable)
    assert restored.metadata.sample_id == "sample-1"
    assert restored.processing_metadata.generated_by.operation == "test_artifact_roundtrip"
    assert np.allclose(restored.to_dataframe()["fraction_frozen"], [0.3])


def test_module_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ufolaf", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "UFOLAF table processing" in result.stdout
