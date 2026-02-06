import yaml

with open("train_outs/epoch_44.yaml", "r") as file:
    data = yaml.safe_load(file)

data["dataset_conf"] |= {
    "apply_random_gain": True,
    "apply_random_gain_conf": {
        "db": [0.125, 2],
        "uniform": True,
        "mode": "amplitude",
    },
}

with open("train_outs/epoch_44.yaml", "w") as file:
    yaml.dump(data, file)