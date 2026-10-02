"""Small synthetic example; run with `python examples/standard_workflow.py`."""

import pandas as pd

import inptk

observations = []
for measurement, counts in (("A_neat", [0, 4, 12]), ("A_diluted", [0, 1, 3])):
    for cycle in ("1", "2"):
        for temperature, frozen in zip((-5, -6, -7), counts):
            observations.append(
                {
                    "measurement_id": measurement,
                    "cycle_id": cycle,
                    "temperature_C": temperature,
                    "n_total": 32,
                    "n_frozen": frozen,
                }
            )

metadata = pd.DataFrame(
    [
        {"measurement_id": "A_neat", "sample_id": "A", "dilution": 1, "droplet_volume_uL": 50},
        {"measurement_id": "A_diluted", "sample_id": "A", "dilution": 10, "droplet_volume_uL": 50},
    ]
)
experiment = inptk.read_counts(pd.DataFrame(observations), metadata=metadata)
result = inptk.analyze_concentration(experiment)

print(result.final.select(group_id="A/1/1").to_dataframe())
# When ready to save, choose a new destination:
# result.save("analysis.inptk")
