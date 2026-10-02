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
result = inptk.analyze_concentration(
    experiment,
    curves={
        "A": {"inputs": ["A_neat", "A_diluted"], "cycle": "1"},
        "A_neat": {"inputs": ["A_neat"], "cycle": "1"},
        "A_diluted": {"inputs": ["A_diluted"], "cycle": "1"},
        "A_cycle2": {"inputs": ["A_neat", "A_diluted"], "cycle": "2"},
    },
)

print(result.curves["A"].cumulative.to_dataframe())
print(result.curves["A"].sources)
# Individual and combined curves use the same interface. They reuse original
# observations; keeping these outputs does not create more independent droplets.
print({name: curve.kind for name, curve in result.curves.items()})
# When ready to save, choose a new destination:
# result.save("analysis.inptk")

# Suggest editable count-based limits before using Average on a selected cycle.
average_curves = {"A": {"inputs": ["A_neat", "A_diluted"], "cycle": "1"}}
suggestions = inptk.suggest_temperature_ranges(experiment, curves=average_curves)
print(suggestions.inputs)  # Includes the reason for each proposed cutoff.
average = inptk.analyze_concentration(
    experiment, curves=average_curves, method="average",
    temperature_ranges_C=suggestions.temperature_ranges_C,
)
print(average.curves["A"].cumulative.to_dataframe())
