## Copyright (c) Microsoft Corporation.
## Licensed under the MIT license.

import datasets
import pickle

def get_dataset(args):
    if args.dataset == "glue/sst2" or args.dataset == "sst2":
        train_dataset = datasets.load_dataset("glue", "sst2")["train"]
        test_dataset = datasets.load_dataset("glue", "sst2")["validation"]
        options = ["negative", "positive"]
        template = "Sentence: {sentence}\nSentiment: {answer}"

        # add options to each example
        train_dataset = train_dataset.map(lambda example: {**example, "options": options})
        test_dataset = test_dataset.map(lambda example: {**example, "options": options})
        
        input_keys = ["sentence"]
        recalibrate_every = False
        balanced_sampling = True
        
    elif args.dataset == "super_glue/wsc" or args.dataset == "wsc":
        train_dataset = datasets.load_dataset("super_glue", "wsc")["train"]
        test_dataset = datasets.load_dataset("super_glue", "wsc")["validation"]
        options = ["no", "yes"]
        template = "Question: In the sentence \"{text}\", does the pronoun '{span2_text}' refer to {span1_text}?\nAnswer: {answer}"

        # add options to each example
        train_dataset = train_dataset.map(lambda example: {**example, "options": options})
        test_dataset = test_dataset.map(lambda example: {**example, "options": options})

        input_keys = ["text"]
        recalibrate_every = True
        balanced_sampling = False
    else:
        raise NotImplementedError

    return {
        "train": train_dataset,
        "test": test_dataset,
        "template": template,
        "input_keys": input_keys,
        "recalibrate_every": recalibrate_every,
        "balanced_sampling": balanced_sampling
    }
