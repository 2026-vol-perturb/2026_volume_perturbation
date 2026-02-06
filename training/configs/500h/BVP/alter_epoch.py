import yaml

with open("train_outs/epoch_44.yaml", "r") as file:
    data = yaml.safe_load(file)

data["dataset_conf"] |= {
    "apply_random_gain": True,
    "apply_random_gain_conf": {
        "db": [0],
        "interval_s": 1.0,
        "vary_db": [-80, -60, -40, -20, 0, 20],
    },
}

with open("train_outs/epoch_44.yaml", "w") as file:
    yaml.dump(data, file)