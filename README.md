# Set up
- With the scaled & filtered CIC23 dataset and a python venv with neccessary packages, run:

```sh
cd data/
python3 CIC23_groupsplit.py
```

This script categorizes the whole dataset into 34 files, then split training/testing from them (more class balance than shuffle the whole thing then split, which is infeasible without NASA-spec RAM and risk not having an entire minor class in train set if you hit that jackpot).

# Partitioning:
- Currently support IID (equal sample distribution per class, per client) and non-IID (label-skewed, which should be naturally quantity skewed of a low entropy). Run:

```sh
python3 partition_data.py --num-clients x --partition-type y # with x = [10|20|30] and y = [iid|label_skew]
```

# Run:

- Runs the main simulation after partitioning:

```sh
python3 fedup.py --n_clients x --partition_type y-x --model_type [dense|gru] --strategy [FedAvg|FedAGRU]
```

- For example:

```sh
python3 fedup.py --n_clients 10 --partition_type iid-10 --strategy FedAGRU --model_type dense
```

- Additional parameters:

```sh
--batch_size: default 8192
--rounds: default 10
--epochs: default 5, local model fit epochs per round
--patience_T: default 3, specific to FedAGRU only
--selection: default true, set client disabling for FedAGRU on or off
```
