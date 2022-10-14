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
    eval_checkpoint = args.eval_checkpoint

    BASE_DIR = "gs://t5_fewshot_mqg" #@param { type: "string" }
    if not BASE_DIR or BASE_DIR == "gs://":
        raise ValueError("You must enter a BASE_DIR.")
    DATA_DIR = os.path.join(BASE_DIR, "data/t5_experiment_1/")
    MODELS_DIR = os.path.join(BASE_DIR, "models/t5_experiment_1/")
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
    fhp_tsv_path = {support_type: {
        "test": os.path.join(DATA_DIR, f"test.{support_type}.tsv")
    } for support_type in ['supp', 'all']}

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

    def fewshot_qg_preprocessor(ds):
        def normalize_text(text):
            """Remove quotes from a TensorFlow string."""
            #text = tf.strings.lower(text)
            text = tf.strings.regex_replace(text,"'(.*)'", r"\1")
            text = tf.strings.regex_replace(text,'"(.*)"', r"\1")
            text = tf.strings.regex_replace(text,"\"", "")
            text = tf.strings.regex_replace(text,"'", "")
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
        save_checkpoints_steps=1000,
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

    # Use a larger batch size for evaluation, which requires less memory.
    model.batch_size = train_batch_size*4

    # model.eval(
    #     mixture_or_task_name="fewshot_mqg_rationale",
    #     checkpoint_steps=eval_checkpoint,
    #     compute_sequence_length=False,
    #     split='validation'
    # )


    # In[53]:


    import csv
    import string 

    # Storing the sets of punctuation in variable result 
    result = string.punctuation 

    test_file = fhp_tsv_path['all']['test']

    with tf.io.gfile.GFile(test_file) as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        test_data = [{"k": k, "p1":p1, "p2":p2, "a": a, "gold_qtype": qtype, "gold_q": gold_q} for k, (p1,p2,a,qtype,gold_q) in enumerate(tsv_reader)]

    # In[54]:


    # from collections import Counter

    # ids = [i for i, e in enumerate(test_data) if not ((e['a'] in e['p1']) or (e['a'] in e['p2'])) and not e['a'] in ('yes', 'no')]
    # for idx in ids:
    #     example = test_data[idx]
    #     print(example['gold_qtype'])
    #     print(example['a'])
    #     print(example['p1'])
    #     print(example['p2'])
    #     print()
    # assert ids == []
    # ids = [i for i, e in enumerate(test_data) if ((e['a'] in e['p1']) and (e['a'] in e['p2']))]
    # print(len(ids))


    # In[60]:


    from collections import defaultdict
    import t5.data.mixtures

    sentinel_token = [f'<extra_id_{num}>' for num in range(100)]

    # Write out the supplied test examples to text files
    predict_inputs_path = {task_num: os.path.join(MODEL_DIR, f"predict_inputs_{task_num}.txt") for task_num in range(1, 14)}
    predict_outputs_path = {task_num: os.path.join(MODEL_DIR, f"predict_outputs_{task_num}.txt") for task_num in range(1, 14)}

    # For each task, example number -> list of input/output file line numbers (0 indexed)
    # Dict of dict -> list of ints
    predict_e2l = {task_num: None for task_num in range(1,14)}

    # For each task, input/output file line number -> example number (0 indexed)
    # Dict of dict -> int 
    predict_l2e = {task_num: None for task_num in range(1,14)}


    # In[78]:


    task_input_prompts = {
        1: lambda i: f"Context 1: {i['p1']} Context 2: {i['p2']} Answer: {i['a']} Common entities found: {sentinel_token[0]} Question type: {sentinel_token[1]}",
        2: lambda i: f"Context 1: {i['p1']} Context 2: {i['p2']} Answer: {i['a']} Common entities found: {sentinel_token[0]} Bridge entity: {sentinel_token[1]}",
        3: lambda i: f"Answer: {i['a']} is {sentinel_token[0]} in context: {i['p']}",
        4: lambda i: f"Entities: {i['a']} and {i['b']} are {sentinel_token[0]}.",
        5: lambda i: f"Context: {i['p']} Bridge entity: {i['b']} Answer: {i['a']} Assertion: {sentinel_token[0]}",
        6: lambda i: f"Context: {i['p']} Bridge entity: {i['b']} Assertion: {sentinel_token[0]}",
        7: lambda i: f"Bridge entity: {i['b']} Assertion 1: {i['s1']} Assertion 2: {i['s2']} Combined: {sentinel_token[0]}",
        8: lambda i: f"Removing bridge entity: {i['b']} from: {i['c']} We get: {sentinel_token[0]}",
        9: lambda i: f"Contract answer entity {i['a']} from: {i['c-b']} We get: {sentinel_token[0]}",
        10: lambda i: f"Turn: {i['c-a']} into question: {sentinel_token[0]}",
        11: lambda i: f"Context 1: {i['p1']} Context 2: {i['p2']} Answer: {i['a']} Assertion from Context 1: {sentinel_token[0]} Assertion from Context 2: {sentinel_token[1]}",
        12: lambda i: f"Assertion 1: {i['s1_c']} Assertion 2: {i['s2_c']} Combine, compare and think: {sentinel_token[0]}",
        13: lambda i: f"Combined assertion: {i['c_c']} Answer: {i['a']} Question: {sentinel_token[0]}",
    }

    class SimpleCounter:
        def __init__(self, count=0):
            self.count = count
        
        def increment(self):
            self.count += 1
        
        def decrement(self):
            self.count -= 1
        
        def get_count(self):
            return self.count
        
        def set_count(self, count):
            self.count = count

    predict_data = test_data

    # Manually apply preprocessing:
    def write_task_input_file(task_num, task_data=predict_data, 
                            predict_inputs_path=predict_inputs_path,
                            predict_outputs_path=predict_outputs_path,
                            predict_e2l=predict_e2l,
                            predict_l2e=predict_l2e):
            
        # dictionary from example number to file line numbers 
        e2l = defaultdict(list)
        
        # dictionary from file line number to example number
        l2e = dict()
        
        # make a counter that keeps track of
        # how many lines have been written a file
        file_lines_written = SimpleCounter(0)
        

        
        with tf.io.gfile.GFile(predict_inputs_path[task_num], "w") as f:
            
            def write_example_line(input_example, e2l=e2l, l2e=l2e, file_obj=f,
                                task_num=task_num, file_lines_written=file_lines_written,
                                task_input_prompts=task_input_prompts):
                input_example_num = input_example['k']
                e2l[input_example_num].append(file_lines_written.get_count())
                l2e[file_lines_written.get_count()] = input_example_num
                file_obj.write(task_input_prompts[task_num](input_example) + "\n")
                file_lines_written.increment()
                
            for k, i in enumerate(task_data):
                if task_num == 1:
                    write_example_line(i)
                elif task_num in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
                    if i['qtype'] in ['bridge', 'confused']:
                        if task_num in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
                            if task_num in [2, 4, 7, 8, 9, 10]:
                                write_example_line(i)

                            elif task_num == 3:
                                i['p'] = i['p1']
                                write_example_line(i)
                                i['p'] = i['p2']
                                write_example_line(i)
                                i.pop('p', None)

                            elif task_num == 5:
                                if not i['b_same_as_a'] or not(i['a_in_p1'] and i['a_in_p2']):
                                    if i['a_in_p1']:
                                        i['p'] = i['p1']
                                        task_data[k]['5_p1'] = True
                                        task_data[k]['5_p2'] = False
                                        write_example_line(i)
                                        i.pop('p', None)
                                    elif i['a_in_p2']:
                                        i['p'] = i['p2']
                                        task_data[k]['5_p2'] = True
                                        task_data[k]['5_p1'] = False
                                        write_example_line(i)
                                        i.pop('p', None)
                                else:
                                    task_data[k]['5_p1'] = False
                                    task_data[k]['5_p2'] = False
                                    
                            elif task_num == 6:
                                if i['b_same_as_a'] or (i['a_in_p1'] and i['a_in_p2']):
                                    i['p'] = i['p1']
                                    task_data[k]['6_p1'] = file_lines_written.get_count()
                                    write_example_line(i)
                                    i.pop('p', None)
                                    
                                    i['p'] = i['p2']
                                    task_data[k]['6_p2'] = file_lines_written.get_count()
                                    write_example_line(i)
                                    i.pop('p', None)
                                else:
                                    if (i['a_in_p1']) and (not i['a_in_p2']):
                                        i['p'] = i['p2']
                                        task_data[k]['6_p2'] = file_lines_written.get_count()
                                        write_example_line(i)
                                        i.pop('p', None)
                                        
                                    if (not i['a_in_p1']) and i['a_in_p2']: 
                                        i['p'] = i['p1']
                                        task_data[k]['6_p1'] = file_lines_written.get_count()
                                        write_example_line(i)
                                        i.pop('p', None)
                                
                                for key in ['6_p1', '6_p2']:
                                    if key not in task_data[k].keys():
                                        task_data[k][key] = None

                    if i['qtype'] in ['comparison', 'confused']:
                        if task_num in [11, 12, 13]:
                            write_example_line(i)
        
        predict_e2l[task_num] = e2l
        predict_l2e[task_num] = l2e
                            
    # TODO for output parsing:
    # add failsafes for tasks 1, 2, 3, 4, 7, 8, 9, 12 
    # only verification possible: 5, 6, 10 
    # only minimal failsafe/verify possible: 11, 13
    def verify(**kwargs):
        return True
    def failsafe(**kwargs):
        return "bleh", "bleh"

    # Manually apply postprocessing and enrich the task_data with outputs from the task output
    def parse_task_output_file(task_num, task_data, predict_e2l, predict_l2e):
        e2l = predict_e2l[task_num]
        l2e = predict_l2e[task_num]
        
        prediction_files = sorted(tf.io.gfile.glob(predict_outputs_path[task_num] + "*"))
        print(f"\nPredictions for task {task_num} using checkpoint %s were read!\n" % prediction_files[-1].split("-")[-1])
        
        with tf.io.gfile.GFile(prediction_files[-1]) as f:
            predict_raw = [line.strip() for line in f.readlines()]
        
        predictions = []
        for j, line in enumerate(predict_raw):  
            
            if task_num in [1,2,11]: # Task 1, 2, 11 have two sentinel tokens
                result = re.search(r"<extra_id_0>(.*)<extra_id_1>(.*)", line)
                result1 = re.search(r"<extra_id_0>(.*)", line)
                result2 = re.search(r"(.*)<extra_id_1>(.*)", line)
                if result:
                    out1, out2 = result.group(1).strip(), result.group(2).strip()
                    if not verify(out_1=out1, out2=out2, task_num=task_num, 
                                example=task_data[l2e[j]], line_num=j, 
                                everything=line, after_sentinel_0=None, after_sentinel_1=None):
                        out1, out2 = failsafe(task_num=task_num, example=task_data[l2e[j]], 
                                            line_num=j, everything=line,
                                            after_sentinel_0=result1, after_sentinel_1=result2)
                else:
                    out1, out2 = failsafe(task_num=task_num, example=task_data[l2e[j]], 
                                            line_num=j, everything=line, 
                                            after_sentinel_0=result1, after_sentinel_1=result2)
            else: # Rest of the tasks only produce one output
                result = re.search(r"<extra_id_0>(.*)", line)
                if result:
                    out1, out2 = result.group(1).strip(), None
                    if not verify(out_1=out1, out2=out2, task_num=task_num, 
                                example=task_data[l2e[j]], line_num=j, 
                                everything=line, after_sentinel_0=None, after_sentinel_1=None):
                        out1, out2 = failsafe(task_num=task_num, example=task_data[l2e[j]], 
                                            line_num=j, everything=line,
                                            after_sentinel_0=None, after_sentinel_1=None)
                        
                else:
                    out1, out2 = failsafe(task_num=task_num, example=task_data[l2e[j]], 
                                        line_num=j, everything=line,
                                        after_sentinel_0=None, after_sentinel_1=None)
            
            predictions.append((out1, out2))
        
        for k, example in enumerate(task_data):
            line_nums = e2l.get(k)
            if line_nums != None:
                if task_num == 1:
                    line_num = line_nums[0]
                    if example['a'] in ('yes', 'no'):
                        task_data[k]['qtype'] = 'comparison'
                    elif 'bridge' in predictions[line_num][1]:
                        task_data[k]['qtype'] = 'bridge'
                    elif 'comparison' in predictions[line_num][1]:
                        task_data[k]['qtype'] = 'comparison'
                    else:
                        task_data[k]['qtype'] = 'confused'

                elif task_num == 2:
                    line_num = line_nums[0]
                    bridge_tentative = re.sub('<extra_id_[0-9]{1,2}>', '', (predictions[line_num][1])).strip()
                    if bridge_tentative != '':
                        task_data[k]['b'] = bridge_tentative
                    else:
                        _, task_data[k]['b'] = failsafe(task_num=task_num,
                                                        example=example,
                                                        line_num=line_num)

                elif task_num == 3:
                    for idx in [0, 1]:
                        line_num = line_nums[idx]
                        p = predictions[line_num][0]
                        if 'present' in p or 'absent' in p:
                            if 'present' in p:
                                task_data[k][f'a_in_p{idx+1}'] = True
                            else:
                                task_data[k][f'a_in_p{idx+1}'] = False
                        else:
                            task_data[k][f'a_in_p{idx+1}'], _ = failsafe(task_num=task_num, 
                                                                        example=example, 
                                                                        line_num=line_num)
                elif task_num == 4:
                    idx = 0
                    line_num = line_nums[idx]
                    p = predictions[line_num][0]
                    if 'similar' in p or 'dissimilar' in p:
                        if 'dissimilar' in p:
                            task_data[k][f'b_same_as_a'] = False
                        else:
                            task_data[k][f'b_same_as_a'] = True
                    else:
                        task_data[k][f'b_same_as_a'] = failsafe(task_num=task_num,
                                                                example=example,
                                                                line_num=line_num)

                elif task_num == 5:
                    idx = 0
                    line_num = line_nums[idx]
                    if task_data[k]['5_p1']:
                        task_data[k]['s1'] = predictions[line_num][0] if predictions[line_num][0] != '' else task_data[k]['p1']
                    elif task_data[k]['5_p2']:
                        task_data[k]['s2'] = predictions[line_num][0] if predictions[line_num][0] != '' else task_data[k]['p2']

                elif task_num == 6:
                    for line_num in line_nums:
                        if '6_p1' in task_data[k].keys():
                            if task_data[k]['6_p1'] == line_num:
                                # task 5 gets priority to write s1, alternate assertion produced by 6th task is saved in key 's1_'
                                if 's1' not in task_data[k].keys():
                                    task_data[k]['s1'] = predictions[line_num][0] if predictions[line_num][0] != '' else task_data[k]['p1']
                                else:
                                    task_data[k]['s1_'] = predictions[line_num][0] if predictions[line_num][0] != '' else task_data[k]['p1']
                        if '6_p2' in task_data[k].keys():
                            if task_data[k]['6_p2'] == line_num:
                                # task 5 gets priority to write s2, alternate assertion produced by 6th task is saved in key 's2_'
                                if 's2' not in task_data[k].keys():
                                    task_data[k]['s2'] = predictions[line_num][0] if predictions[line_num][0] != '' else task_data[k]['p2']
                                else:
                                    task_data[k]['s2_'] = predictions[line_num][0] if predictions[line_num][0] != '' else task_data[k]['p2']
                elif task_num == 7:
                    idx = 0
                    line_num = line_nums[idx]
                    task_data[k]['c'] = predictions[line_num][0] if predictions[line_num][0] != '' else f"{task_data[k]['s1']}. {task_data[k]['s2']}"
                        
                elif task_num == 8:
                    idx = 0
                    line_num = line_nums[idx]
                    task_data[k]['c-b'] = predictions[line_num][0] if predictions[line_num][0] != '' else task_data[k]['c'].replace(task_data[k]['b'], '')
                    
                elif task_num == 9:
                    idx = 0
                    line_num = line_nums[idx]
                    task_data[k]['c-a'] = predictions[line_num][0] if predictions[line_num][0] != '' else task_data[k]['c-b'].replace(task_data[k]['a'], 'certain entity')
                    
                elif task_num == 10:
                    idx = 0
                    line_num = line_nums[idx]
                    task_data[k]['q'] = predictions[line_num][0] if predictions[line_num][0] != '' else task_data[k]['c-a'].replace('certain', 'what') + ' ?'
                    
                elif task_num == 11:
                    idx = 0
                    line_num = line_nums[idx]
                    task_data[k]['s1_c'] = predictions[line_num][0] if predictions[line_num][0] != '' else task_data[k]['p1']
                    task_data[k]['s2_c'] = predictions[line_num][1] if predictions[line_num][1] != '' else task_data[k]['p2']
                    
                elif task_num == 12:
                    idx = 0
                    line_num = line_nums[idx]
                    task_data[k]['c_c'] = predictions[line_num][0] if predictions[line_num][0] != '' else task_data[k]['s1_c'] + ' ' + task_data[k]['s1_c']
                    
                elif task_num == 13:
                    idx = 0
                    line_num = line_nums[idx]
                    task_data[k]['q_c'] = predictions[line_num][0] if predictions[line_num][0] != '' else task_data[k]['s1_c'] + ' ' + task_data[k]['s1_c']     
        return

    # In[82]:
    model.batch_size = 64

    for task_num in range(1,14):
        write_task_input_file(task_num)
        model.predict(
        input_file=predict_inputs_path[task_num],
        output_file=predict_outputs_path[task_num],
        checkpoint_steps=eval_checkpoint,
        temperature=0,
        )
        parse_task_output_file(task_num=task_num, task_data=predict_data, predict_e2l=predict_e2l, predict_l2e=predict_l2e)
    
    targets = []
    predictions = []
    for i in range(len(predict_data)):
        targets.append(predict_data[i]['gold_q'])
        predicted_questions = []
        for qk in ['q', 'q_c']: 
            if qk in predict_data[i].keys():
                predicted_questions.append(predict_data[i][qk])
        predictions.append(predict)
    fin = time.time()


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
    parser.add_argument('--eval_checkpoint', type=int, required=True,
                        help='checkpoint step number to do evaluation on')
    
    args = parser.parse_args()
    main(args)