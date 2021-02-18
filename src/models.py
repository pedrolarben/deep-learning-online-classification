def get_models_params():
    models = {
        "MLP": {"hidden_layers": [32, 64, 128], "activation": "relu", "dropout": 0.2},
        "LSTM": {
            "recurrent_units": [64, 128],
            "recurrent_dropout": 0,
            "return_sequences": True,
            "dense_layers": [64, 32],
            "dense_dropout": 0.2,
            "dense_activation": "relu",
        },
        "CNN": {
            "conv_layers": [64, 128],
            "kernel_sizes": [7, 5],
            "pool_sizes": [2, 2],
            "dense_layers": [64, 32],
            "activation": "relu",
            "dense_activation": "relu",
            "dense_dropout": 0.2,
        },
        "TCN": {
            "nb_filters": 64,
            "kernel_size": 5,
            "nb_stacks": 1,
            "dilations": [1, 2, 4, 8, 16, 64],
            "tcn_dropout": 0.0,
            "return_sequences": True,
            "activation": "relu",
            "padding": "causal",
            "use_skip_connections": True,
            "use_batch_norm": False,
            "dense_layers": [64, 32],
            "dense_dropout": 0.2,
            "dense_activation": "relu",
        },
    }
    return models
