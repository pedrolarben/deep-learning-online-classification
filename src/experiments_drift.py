import os
import ADLStream
from tqdm import tqdm
from datasets import get_concept_drift_datasets
from models import get_models_params

N = 500000
STREAM_PERIOD = 10

PREQ_CHUNK = 10
PREQ_FADINGFACTOR = 0.98
PREQ_METRIC = "kappa"

BATCH_SIZE = 32
N_BATCHES_FED = 10

concept_drift_datasets = get_concept_drift_datasets(N)
models = get_models_params()

for dataset_name in tqdm(concept_drift_datasets, desc="dataset"):
    for model_name in tqdm(models, desc="model"):

        dataset = concept_drift_datasets[dataset_name]["dataset"]
        onehot_classes = concept_drift_datasets[dataset_name]["onehot_classes"]
        preprocessing_pipeline = concept_drift_datasets[dataset_name]["preprocessing"]

        stream = ADLStream.data.stream.RiverStream(
            dataset, n_instances=N, stream_period=STREAM_PERIOD
        )

        stream_generator = ADLStream.data.ClassificationStreamGenerator(
            stream=stream,
            one_hot_labels=onehot_classes,
            preprocessing_steps=preprocessing_pipeline,
            max_instances=N,
        )

        result_file = f"../results/drift/{model_name}"
        if not os.path.isdir(result_file):
            os.makedirs(result_file)
        result_file = f"{result_file}/{dataset_name}.csv"

        evaluator = ADLStream.evaluation.PrequentialEvaluator(
            chunk_size=PREQ_CHUNK,
            metric=PREQ_METRIC,
            fadding_factor=PREQ_FADINGFACTOR,
            results_file=result_file,
            show_plot=False,
        )

        model_parameters = models[model_name]
        model_parameters["out_activation"] = (
            "sigmoid"
            if onehot_classes is None or len(onehot_classes) == 2
            else "softmax"
        )
        model_loss = (
            "binary_crossentropy"
            if onehot_classes is None or len(onehot_classes) == 2
            else "categorical_crossentropy"
        )
        model_optimizer = "adam"

        adls = ADLStream.ADLStream(
            stream_generator=stream_generator,
            evaluator=evaluator,
            batch_size=BATCH_SIZE,
            num_batches_fed=N_BATCHES_FED,
            model_architecture=model_name,
            model_loss=model_loss,
            model_optimizer=model_optimizer,
            model_parameters=model_parameters,
            log_file="ADLStream.log",
        )

        adls.run()
