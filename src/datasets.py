import os
import pandas

from river import synth

from ADLStream.data.preprocessing import MinMaxScaler
from ADLStream.data.stream import CSVFileStream


def _get_class_values(dataset_path):
    vals = set()
    with open(dataset_path, "r") as f:
        for line in f:
            val = int(line.split(",")[-1])
            vals.add(val)
    return vals


def get_time_series_datasets():
    datadir = "../data/"
    suffix = ".csv"
    dataset_list = [
        s.replace(suffix, "") for s in os.listdir(datadir) if s.endswith(suffix)
    ]

    datasets = {}
    for dataset_name in dataset_list:
        dataset_path = f"{datadir}{dataset_name}{suffix}"
        classes = _get_class_values(dataset_path)
        datasets[dataset_name] = {
            "dataset": dataset_path,
            "onehot_classes": list(classes),
            "preprocessing": [MinMaxScaler(share_params=True)],
        }

    return datasets


def get_concept_drift_datasets(n_instances):
    concept_drift_stream = {
        "RBFi-slow": {
            "dataset": synth.RandomRBFDrift(
                change_speed=0.0001, n_features=20, n_classes=3
            ),
            "onehot_classes": [0, 1, 2],
            "drift_type": "incremental",
            "preprocessing": [],
        },
        "RBFi-fast": {
            "dataset": synth.RandomRBFDrift(
                change_speed=0.001, n_features=20, n_classes=3
            ),
            "onehot_classes": [0, 1, 2],
            "drift_type": "incremental",
            "preprocessing": [],
        },
        "LED-4": {
            "dataset": synth.LEDDrift(
                n_drift_features=4, noise_percentage=0.1, irrelevant_features=True
            ),
            "onehot_classes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "drift_type": "incremental",
            "preprocessing": [],
        },
        "RTGa": {
            "dataset": synth.ConceptDriftStream(
                stream=synth.RandomTree(
                    seed_tree=1,
                    seed_sample=1,
                    n_classes=3,
                    n_num_features=20,
                    n_cat_features=0,
                    max_tree_depth=7,
                    first_leaf_level=5,
                ),
                drift_stream=synth.RandomTree(
                    seed_tree=2,
                    seed_sample=2,
                    n_classes=3,
                    n_num_features=20,
                    n_cat_features=0,
                    max_tree_depth=5,
                    first_leaf_level=3,
                ),
                position=int(n_instances / 2),
                width=1,
            ),
            "onehot_classes": [0, 1, 2],
            "drift_type": "abrupt",
            "preprocessing": [MinMaxScaler()],
        },
        "RTGa3": {
            "dataset": synth.ConceptDriftStream(
                stream=synth.RandomTree(
                    seed_tree=1,
                    seed_sample=1,
                    n_classes=3,
                    n_num_features=20,
                    n_cat_features=0,
                    max_tree_depth=7,
                    first_leaf_level=5,
                ),
                drift_stream=synth.ConceptDriftStream(
                    stream=synth.RandomTree(
                        seed_tree=2,
                        seed_sample=2,
                        n_classes=3,
                        n_num_features=20,
                        n_cat_features=0,
                        max_tree_depth=7,
                        first_leaf_level=3,
                    ),
                    drift_stream=synth.ConceptDriftStream(
                        stream=synth.RandomTree(
                            seed_tree=3,
                            seed_sample=3,
                            n_classes=3,
                            n_num_features=20,
                            n_cat_features=0,
                            max_tree_depth=5,
                            first_leaf_level=5,
                        ),
                        drift_stream=synth.RandomTree(
                            seed_tree=4,
                            seed_sample=4,
                            n_classes=3,
                            n_num_features=20,
                            n_cat_features=0,
                            max_tree_depth=5,
                            first_leaf_level=3,
                        ),
                        position=int(n_instances / 4),
                        width=1,
                    ),
                    position=int(n_instances / 4),
                    width=1,
                ),
                position=int(n_instances / 4),
                width=1,
            ),
            "onehot_classes": [0, 1, 2],
            "dift_type": "abrupt",
            "preprocessing": [MinMaxScaler()],
        },
        "RTGa6": {
            "dataset": synth.ConceptDriftStream(
                stream=synth.RandomTree(
                    seed_tree=1,
                    seed_sample=1,
                    n_classes=3,
                    n_num_features=20,
                    n_cat_features=0,
                    max_tree_depth=7,
                    first_leaf_level=5,
                ),
                drift_stream=synth.ConceptDriftStream(
                    stream=synth.RandomTree(
                        seed_tree=2,
                        seed_sample=2,
                        n_classes=3,
                        n_num_features=20,
                        n_cat_features=0,
                        max_tree_depth=7,
                        first_leaf_level=3,
                    ),
                    drift_stream=synth.ConceptDriftStream(
                        stream=synth.RandomTree(
                            seed_tree=3,
                            seed_sample=3,
                            n_classes=3,
                            n_num_features=20,
                            n_cat_features=0,
                            max_tree_depth=5,
                            first_leaf_level=5,
                        ),
                        drift_stream=synth.ConceptDriftStream(
                            stream=synth.RandomTree(
                                seed_tree=3,
                                seed_sample=3,
                                n_classes=3,
                                n_num_features=20,
                                n_cat_features=0,
                                max_tree_depth=7,
                                first_leaf_level=5,
                            ),
                            drift_stream=synth.ConceptDriftStream(
                                stream=synth.RandomTree(
                                    seed_tree=3,
                                    seed_sample=3,
                                    n_classes=3,
                                    n_num_features=20,
                                    n_cat_features=0,
                                    max_tree_depth=7,
                                    first_leaf_level=3,
                                ),
                                drift_stream=synth.ConceptDriftStream(
                                    stream=synth.RandomTree(
                                        seed_tree=3,
                                        seed_sample=3,
                                        n_classes=3,
                                        n_num_features=20,
                                        n_cat_features=0,
                                        max_tree_depth=5,
                                        first_leaf_level=5,
                                    ),
                                    drift_stream=synth.RandomTree(
                                        seed_tree=4,
                                        seed_sample=4,
                                        n_classes=3,
                                        n_num_features=20,
                                        n_cat_features=0,
                                        max_tree_depth=7,
                                        first_leaf_level=5,
                                    ),
                                    position=int(n_instances / 7),
                                    width=1,
                                ),
                                position=int(n_instances / 7),
                                width=1,
                            ),
                            position=int(n_instances / 7),
                            width=1,
                        ),
                        position=int(n_instances / 7),
                        width=1,
                    ),
                    position=int(n_instances / 7),
                    width=1,
                ),
                position=int(n_instances / 7),
                width=1,
            ),
            "onehot_classes": [0, 1, 2],
            "dift_type": "abrupt",
            "preprocessing": [MinMaxScaler()],
        },
        "ARGWa-F1F4": {
            "dataset": synth.ConceptDriftStream(
                stream=synth.Agrawal(classification_function=1),
                drift_stream=synth.Agrawal(classification_function=4),
                position=int(n_instances / 2),
                width=1,
            ),
            "onehot_classes": None,
            "drift_type": "abrupt",
            "preprocessing": [MinMaxScaler()],
        },
        "ARGWa-F2F5F8": {
            "dataset": synth.ConceptDriftStream(
                stream=synth.Agrawal(classification_function=2),
                drift_stream=synth.ConceptDriftStream(
                    stream=synth.Agrawal(classification_function=5),
                    drift_stream=synth.Agrawal(classification_function=8),
                    position=int(n_instances / 3),
                    width=1,
                ),
                position=int(n_instances / 3),
                width=1,
            ),
            "onehot_classes": None,
            "drift_type": "abrupt",
            "preprocessing": [MinMaxScaler()],
        },
        "SEAa-F2F4": {
            "dataset": synth.ConceptDriftStream(
                stream=synth.SEA(variant=1, noise=0.02),
                drift_stream=synth.SEA(variant=2, noise=0.02),
                position=int(n_instances / 2),
                width=1,
            ),
            "onehot_classes": None,
            "drift_type": "abrupt",
            "preprocessing": [MinMaxScaler()],
        },
        "RTGg": {
            "dataset": synth.ConceptDriftStream(
                stream=synth.RandomTree(
                    seed_tree=1,
                    seed_sample=1,
                    n_classes=3,
                    n_num_features=20,
                    n_cat_features=0,
                    max_tree_depth=7,
                    first_leaf_level=5,
                ),
                drift_stream=synth.RandomTree(
                    seed_tree=2,
                    seed_sample=2,
                    n_classes=3,
                    n_num_features=20,
                    n_cat_features=0,
                    max_tree_depth=5,
                    first_leaf_level=3,
                ),
                position=int(n_instances / 2),
                width=int(n_instances / 10),
            ),
            "onehot_classes": [0, 1, 2],
            "drift_type": "abrupt",
            "preprocessing": [MinMaxScaler()],
        },
        "RTGg3": {
            "dataset": synth.ConceptDriftStream(
                stream=synth.RandomTree(
                    seed_tree=1,
                    seed_sample=1,
                    n_classes=3,
                    n_num_features=20,
                    n_cat_features=0,
                    max_tree_depth=7,
                    first_leaf_level=5,
                ),
                drift_stream=synth.ConceptDriftStream(
                    stream=synth.RandomTree(
                        seed_tree=2,
                        seed_sample=2,
                        n_classes=3,
                        n_num_features=20,
                        n_cat_features=0,
                        max_tree_depth=7,
                        first_leaf_level=3,
                    ),
                    drift_stream=synth.ConceptDriftStream(
                        stream=synth.RandomTree(
                            seed_tree=3,
                            seed_sample=3,
                            n_classes=3,
                            n_num_features=20,
                            n_cat_features=0,
                            max_tree_depth=5,
                            first_leaf_level=5,
                        ),
                        drift_stream=synth.RandomTree(
                            seed_tree=4,
                            seed_sample=4,
                            n_classes=3,
                            n_num_features=20,
                            n_cat_features=0,
                            max_tree_depth=5,
                            first_leaf_level=3,
                        ),
                        position=int(n_instances / 4),
                        width=int(n_instances / 10),
                    ),
                    position=int(n_instances / 4),
                    width=int(n_instances / 10),
                ),
                position=int(n_instances / 4),
                width=int(n_instances / 10),
            ),
            "onehot_classes": [0, 1, 2],
            "dift_type": "abrupt",
            "preprocessing": [MinMaxScaler()],
        },
        "RTGg6": {
            "dataset": synth.ConceptDriftStream(
                stream=synth.RandomTree(
                    seed_tree=1,
                    seed_sample=1,
                    n_classes=3,
                    n_num_features=20,
                    n_cat_features=0,
                    max_tree_depth=7,
                    first_leaf_level=5,
                ),
                drift_stream=synth.ConceptDriftStream(
                    stream=synth.RandomTree(
                        seed_tree=2,
                        seed_sample=2,
                        n_classes=3,
                        n_num_features=20,
                        n_cat_features=0,
                        max_tree_depth=7,
                        first_leaf_level=3,
                    ),
                    drift_stream=synth.ConceptDriftStream(
                        stream=synth.RandomTree(
                            seed_tree=3,
                            seed_sample=3,
                            n_classes=3,
                            n_num_features=20,
                            n_cat_features=0,
                            max_tree_depth=5,
                            first_leaf_level=5,
                        ),
                        drift_stream=synth.ConceptDriftStream(
                            stream=synth.RandomTree(
                                seed_tree=3,
                                seed_sample=3,
                                n_classes=3,
                                n_num_features=20,
                                n_cat_features=0,
                                max_tree_depth=7,
                                first_leaf_level=5,
                            ),
                            drift_stream=synth.ConceptDriftStream(
                                stream=synth.RandomTree(
                                    seed_tree=3,
                                    seed_sample=3,
                                    n_classes=3,
                                    n_num_features=20,
                                    n_cat_features=0,
                                    max_tree_depth=7,
                                    first_leaf_level=3,
                                ),
                                drift_stream=synth.ConceptDriftStream(
                                    stream=synth.RandomTree(
                                        seed_tree=3,
                                        seed_sample=3,
                                        n_classes=3,
                                        n_num_features=20,
                                        n_cat_features=0,
                                        max_tree_depth=5,
                                        first_leaf_level=5,
                                    ),
                                    drift_stream=synth.RandomTree(
                                        seed_tree=4,
                                        seed_sample=4,
                                        n_classes=3,
                                        n_num_features=20,
                                        n_cat_features=0,
                                        max_tree_depth=7,
                                        first_leaf_level=5,
                                    ),
                                    position=int(n_instances / 7),
                                    width=int(n_instances / 20),
                                ),
                                position=int(n_instances / 7),
                                width=int(n_instances / 20),
                            ),
                            position=int(n_instances / 7),
                            width=int(n_instances / 20),
                        ),
                        position=int(n_instances / 7),
                        width=int(n_instances / 20),
                    ),
                    position=int(n_instances / 7),
                    width=int(n_instances / 20),
                ),
                position=int(n_instances / 7),
                width=int(n_instances / 20),
            ),
            "onehot_classes": [0, 1, 2],
            "dift_type": "gradual",
            "preprocessing": [MinMaxScaler()],
        },
        "ARGWg-F1F4": {
            "dataset": synth.ConceptDriftStream(
                stream=synth.Agrawal(classification_function=1),
                drift_stream=synth.Agrawal(classification_function=4),
                position=int(n_instances / 2),
                width=int(n_instances / 10),
            ),
            "onehot_classes": None,
            "drift_type": "abrupt",
            "preprocessing": [MinMaxScaler()],
        },
        "ARGWg-F2F5F8": {
            "dataset": synth.ConceptDriftStream(
                stream=synth.Agrawal(classification_function=2),
                drift_stream=synth.ConceptDriftStream(
                    stream=synth.Agrawal(classification_function=5),
                    drift_stream=synth.Agrawal(classification_function=8),
                    position=int(n_instances / 3),
                    width=1,
                ),
                position=int(n_instances / 3),
                width=int(n_instances / 10),
            ),
            "onehot_classes": None,
            "drift_type": "abrupt",
            "preprocessing": [MinMaxScaler()],
        },
        "SEAg-F2F4": {
            "dataset": synth.ConceptDriftStream(
                stream=synth.SEA(variant=1, noise=0.02),
                drift_stream=synth.SEA(variant=2, noise=0.02),
                position=int(n_instances / 2),
                width=int(n_instances / 10),
            ),
            "onehot_classes": None,
            "drift_type": "abrupt",
            "preprocessing": [MinMaxScaler()],
        },
    }

    return concept_drift_stream
