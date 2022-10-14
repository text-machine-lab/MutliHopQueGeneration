import os
import re
import sys
import time
import warnings
import argparse
import functools

from functools import partial

warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

import t5
import t5.models
import seqio

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from statistics import mean
import pandas as pd

def main(args):
    tpu_name = args.tpu_name
    input_len = args.input_len
    target_len = args.target_len
    lr = args.lr
    shot = args.shots
    support = args.support
    MODEL_SIZE = args.model_size

    BASE_DIR = "gs://t5_fewshot_mqg" #@param { type: "string" }
    if not BASE_DIR or BASE_DIR == "gs://":
        raise ValueError("You must enter a BASE_DIR.")
    DATA_DIR = os.path.join(BASE_DIR, "data/t5_experiment_2/")
    MODELS_DIR = os.path.join(BASE_DIR, "models/t5_experiment_2/")
    ON_CLOUD = True

    # Public GCS path for T5 pre-trained model checkpoints
    BASE_PRETRAINED_DIR = "gs://t5-data/pretrained_models"
    PRETRAINED_DIR = os.path.join(BASE_PRETRAINED_DIR, MODEL_SIZE)
    MODEL_DIR = os.path.join(MODELS_DIR, MODEL_SIZE, f"{shot}_SHOT", support)

    model_parallelism, train_batch_size, keep_checkpoint_max = {
        "small": (1, 256,  10),
        "base":  (2, 128,  10),
        "large": (8,  64,  10),
        "3B":    (8,  16,  10),
        "11B":   (8,   4,  10)}[MODEL_SIZE]
    
    # Directory of Fewshot HotpotQG data on GCS.
    fhp_tsv_path = {t: {support_type: {shots: {
        "train": os.path.join(DATA_DIR, f"train.{support_type}.{t}_task.{shots}_shot.tsv"),
        "validation": os.path.join(DATA_DIR, f"dev.{support_type}.{t}_task.{shots}_shot.tsv"),
    } for shots in ['8','16','32','64','128',"all"]} for support_type in ['supp', 'all']} for t in range(1,14)}

    if ON_CLOUD:
        print("Setting up GCS access...")
    # Use legacy GCS authentication method.
    os.environ['USE_AUTH_EPHEM'] = '0'
    import tensorflow_gcs_config
    
    # from google.colab import auth
    # auth.authenticate_user()

    # Set credentials for GCS reading/writing from Colab and TPU.
    TPU_TOPOLOGY = "v3-8"
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)  # TPU detection
        TPU_ADDRESS = tpu.get_master()
        print('Running on TPU:', TPU_ADDRESS)
    except ValueError:
        raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')
    tf.enable_eager_execution()
    tf.config.experimental_connect_to_host(TPU_ADDRESS)
    # tensorflow_gcs_config.configure_gcs_from_colab_auth()

    tf.disable_v2_behavior()

    # Improve logging.
    from contextlib import contextmanager
    import logging as py_logging

    if ON_CLOUD:
        tf.get_logger().propagate = False
    py_logging.root.setLevel('INFO')

    def tf_verbosity_level(level):
        og_level = tf.logging.get_verbosity()
        tf.logging.set_verbosity(level)
        yield
        tf.logging.set_verbosity(og_level)

    def fhp_task_dataset_fn_getter_fn(task_number, support=support, shot=shot):
        def fhp_dataset_fn(split, shuffle_files=False):
            # We only have one file for each split.
            del shuffle_files
            # Load lines from the text file as examples.
            ds = tf.data.TextLineDataset(fhp_tsv_path[task_number][support][shot][split])
            # Split each "<question>\t<answer>" example into (question, answer) tuple.
            ds = ds.map(
                functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                                    field_delim="\t", use_quote_delim=False),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)
            # Map each tuple to a {"question": ... "answer": ...} dict.
            ds = ds.map(lambda *ex: dict(zip(["question", "answer"], ex)))
            return ds
        return fhp_dataset_fn

    print("A few raw validation examples...")
    for ex in tfds.as_numpy(fhp_task_dataset_fn_getter_fn(task_number=1)("validation").take(5)):
        print(ex)

    def fewshot_qg_preprocessor(ds):
        def normalize_text(text):
            """Remove quotes from a TensorFlow string."""
            #text = tf.strings.lower(text)
            text = tf.strings.regex_replace(text,"'(.*)'", r"\1")
            text = tf.strings.regex_replace(text,'"(.*)"', r"\1")
            text = tf.strings.regex_replace(text,"\"", "")
            return text

        def to_inputs_and_targets(ex):
            """Map {"question": ..., "answer": ...}->{"inputs": ..., "targets": ...}."""
            return {
                "inputs":
                    tf.strings.join(
                        [normalize_text(ex["question"])]),
                "targets": normalize_text(ex["answer"])
            }
        return ds.map(to_inputs_and_targets, 
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    DEFAULT_OUTPUT_FEATURES = {
        "inputs":
            seqio.Feature(
                vocabulary=t5.data.get_default_vocabulary(), add_eos=True),
        "targets":
            seqio.Feature(
                vocabulary=t5.data.get_default_vocabulary(), add_eos=True)
    }

    b4m, mm, rm = Bleu(4), Meteor(), Rouge()

    bleu4 = lambda targets, predictions: {'bleu4': 100 * mean([b4m.compute_score({i:[targets[i]]}, {i:[predictions[i]]})[0][3] for i in range(len(targets))])}
    bleu3 = lambda targets, predictions: {'bleu3': 100 * mean([b4m.compute_score({i:[targets[i]]}, {i:[predictions[i]]})[0][2] for i in range(len(targets))])}
    bleu2 = lambda targets, predictions: {'bleu2': 100 * mean([b4m.compute_score({i:[targets[i]]}, {i:[predictions[i]]})[0][1] for i in range(len(targets))])}
    bleu1 = lambda targets, predictions: {'bleu1': 100 * mean([b4m.compute_score({i:[targets[i]]}, {i:[predictions[i]]})[0][0] for i in range(len(targets))])}
    
    rouge = lambda targets, predictions: {'rougeL': 100 * mean([rm.compute_score({i:[targets[i]]}, {i:[predictions[i]]})[0] for i in range(len(targets))])}
    
    def meteor(targets, predictions):
        from pycocoevalcap.meteor.meteor import Meteor
        return {'meteor': 100 * mean([mm.compute_score({i:[targets[i]]}, {i:[predictions[i]]})[0] for i in range(len(targets))])}

    def remove_tags_and_strip(string, **unused_kwargs): 
        return re.sub('<extra_id_[0-9]{1,2}>', '', string).strip()
    
    def get_task_len(task_number,support,shot,split):
        return len(pd.read_csv(fhp_tsv_path[task_number][support][shot][split], sep='\t'))
    
    all_task_examples_numbers = sum([get_task_len(task_number,support,shot,split="train") for task_number in range(1,14)])
    
    FINETUNE_STEPS = max(5000, 35 * all_task_examples_numbers)
    
    print(f"Number of total finetune steps set to {FINETUNE_STEPS}!")
    
    for task_number in range(1,14):
        seqio.TaskRegistry.add(
            f"fhpqg_rationale_task_{task_number}",
            # Specify the task source.
            source=seqio.FunctionDataSource(
                # Supply a function which returns a tf.data.Dataset.
                dataset_fn=fhp_task_dataset_fn_getter_fn(task_number=task_number, support=support, shot=shot),
                splits=["train", "validation"],
                # Not required, but helps for mixing and auto-caching.
                num_input_examples={
                        "train": get_task_len(task_number,support,shot,split="train"), 
                        "validation": get_task_len(task_number,support,shot,split="validation"),
                    }
                ),
            # Supply a list of functions that preprocess the input tf.data.Dataset.
            preprocessors=[
                fewshot_qg_preprocessor,
                seqio.preprocessors.tokenize_and_append_eos,
            ],
            # Lowercase targets before computing metrics.
            # postprocess_fn=remove_tags_and_strip,
            # We'll use accuracy as our evaluation metric.
            metric_fns=[bleu4, bleu3, bleu2, bleu1, meteor, rouge, t5.evaluation.metrics.accuracy],
            output_features=DEFAULT_OUTPUT_FEATURES,
        )

        fhpqg_task = seqio.TaskRegistry.get(f"fhpqg_rationale_task_{task_number}")
        ds = fhpqg_task.get_dataset(split="validation", sequence_length={"inputs": input_len, "targets": target_len})
        print("A few preprocessed validation examples...")
        for ex in tfds.as_numpy(ds.take(2)):
            print(ex)
    
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print(f"Number of total finetune steps set to {FINETUNE_STEPS}!")
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    seqio.MixtureRegistry.remove("fewshot_mqg_rationale")
    seqio.MixtureRegistry.add(
        "fewshot_mqg_rationale",
        [f"fhpqg_rationale_task_{task_number}" for task_number in range(1,14)],
        default_rate=partial(seqio.mixing_rate_num_examples, temperature=3)
    )

    tf.io.gfile.makedirs(MODEL_DIR)
    # The models from our paper are based on the Mesh Tensorflow Transformer.
    model = t5.models.MtfModel(
        model_dir=MODEL_DIR,
        tpu=TPU_ADDRESS,
        tpu_topology=TPU_TOPOLOGY,
        model_parallelism=model_parallelism,
        batch_size=train_batch_size,
        sequence_length={"inputs": input_len, "targets": target_len},
        learning_rate_schedule=lr,
        save_checkpoints_steps=max(FINETUNE_STEPS//10, 1000),
        keep_checkpoint_max=keep_checkpoint_max if ON_CLOUD else None,
        iterations_per_loop=100,
    )

    import sys
    from absl import app

    # Addresses `UnrecognizedFlagError: Unknown command line flag 'f'`
    sys.argv = sys.argv[:1]

    # `app.run` calls `sys.exit`
    try:
        app.run(lambda argv: None)
    except:
        pass

    model.finetune(
        mixture_or_task_name="fewshot_mqg_rationale",
        pretrained_model_dir=PRETRAINED_DIR,
        finetune_steps=FINETUNE_STEPS
    )

    # Use a larger batch size for evaluation, which requires less memory.
    model.batch_size = 4*train_batch_size

    model.eval(
        mixture_or_task_name="fewshot_mqg_rationale",
        checkpoint_steps="all",
        compute_sequence_length=False,
        split='validation'
    )

    # model.eval(
    #     mixture_or_task_name="fewshot_mqg",
    #     checkpoint_steps=1000000 + FINETUNE_STEPS,
    #     compute_sequence_length=False,
    #     split='test'
    # )

if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()
    parser.add_argument('--shots', type=str, required=True,
                        help='Number of shots to train as well as validate with, options are 8, 16, 32, 64, 128')
    parser.add_argument('--model_size', type=str, required=True,
                        help='3B, 11B, large, small, base')
    parser.add_argument('--support', type=str, required=True,
                        help='supporting sentences only: "supp", entire context as input: "all"')
    parser.add_argument('--input_len', type=int, required=True,
                        help='Length of input in num tokens')
    parser.add_argument('--target_len', type=int, required=True,
                        help='Length of target in num tokens')
    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate')
    parser.add_argument('--tpu_name', type=str, required=True,
                        help='name of the tpu_node')

    args = parser.parse_args()
    main(args)