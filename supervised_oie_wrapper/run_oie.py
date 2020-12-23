""" Usage:
    <file-name> --in=INPUT_FILE --batch-size=BATCH-SIZE --out=OUTPUT_FILE [--cuda-device=CUDA_DEVICE] [--debug]
"""
# External imports
import logging
from pprint import pprint
from pprint import pformat
from docopt import docopt
import json
import pdb
from tqdm import tqdm
from allennlp.pretrained import open_information_extraction_stanovsky_2018
from collections import defaultdict
from operator import itemgetter
import functools
import operator
import torch

model_oie = open_information_extraction_stanovsky_2018()
model_oie._model.cuda(0)


# Local imports
from supervised_oie_wrapper.format_oie import format_extractions, Mock_token
#=-----

def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]

def create_instances(model, sent):
    """
    Convert a sentence into a list of instances.
    """
    sent_tokens = model._tokenizer.tokenize(sent)

    # Find all verbs in the input sentence
    pred_ids = [i for (i, t) in enumerate(sent_tokens)
                if t.pos_ == "VERB" or t.pos_ == "AUX"]

    # Create instances
    instances = [{"sentence": sent_tokens,
                  "predicate_index": pred_id}
                 for pred_id in pred_ids]

    return instances

def get_confidence(model, tag_per_token, class_probs):
    """
    Get the confidence of a given model in a token list, using the class probabilities
    associated with this prediction.
    """
    token_indexes = [model._model.vocab.get_token_index(tag, namespace = "labels") for tag in tag_per_token]

    # Get probability per tag
    probs = [class_prob[token_index] for token_index, class_prob in zip(token_indexes, class_probs)]

    # Combine (product)
    prod_prob = functools.reduce(operator.mul, probs)

    return prod_prob

def run_oie(lines, batch_size=64, cuda_device=-1, debug=False):
    """
    Run the OIE model and process the output.
    """

    if debug:
        logging.basicConfig(level = logging.DEBUG)
    else:
        logging.basicConfig(level = logging.INFO)

    # Init OIE
    model = model_oie
    # model = open_information_extraction_stanovsky_2018()
    #
    #
    # # Move model to gpu, if requested
    # if cuda_device >= 0:
    #     model._model.cuda(cuda_device)


    # process sentences
    logging.info("Processing sentences")
    oie_lines = []
    oie_lines_dict = []
    for chunk in tqdm(chunks(lines, batch_size)):
        oie_inputs = []
        sentTokensList = []
        for sent_idx ,sent in enumerate(chunk):
            # if len(sent) > 20000: #if sentence is too long for memory (probably garbage sentence)
            #     sent = ''
            pred_instance = create_instances(model, sent)
            oie_inputs.extend(pred_instance)

            sentTokensList.append(" ".join([str(token) for token in model._tokenizer.tokenize(sent)]))


        # Run oie on sents
        sent_preds = []
        if oie_inputs:
            sent_preds = model.predict_batch_json(oie_inputs)

        # Collect outputs in batches
        predictions_by_sent = defaultdict(list)
        for outputs in sent_preds:
            sent_tokens = outputs["words"]
            tags = outputs["tags"]
            sent_str = " ".join(sent_tokens)
            assert(len(sent_tokens) == len(tags))
            predictions_by_sent[sent_str].append((outputs["tags"], outputs["class_probabilities"]))


        # Create extractions by sentence
        for sent_tokens in sentTokensList:
            if sent_tokens not in predictions_by_sent:    # handle sentences without predicate
                oie_lines.extend([None])
                oie_lines_dict.extend([None])
                continue
            predictions_for_sent = predictions_by_sent[sent_tokens]
            raw_tags = list(map(itemgetter(0), predictions_for_sent))
            class_probs = list(map(itemgetter(1), predictions_for_sent))

            # Compute confidence per extraction
            confs = [get_confidence(model, tag_per_token, class_prob)
                     for tag_per_token, class_prob in zip(raw_tags, class_probs)]

            extractions, tags, results_dict = format_extractions([Mock_token(tok) for tok in sent_tokens.split(" ")], raw_tags)

            oie_lines.extend([extraction + f"\t{conf}" for extraction, conf in zip(extractions, confs)])
            oie_lines_dict.extend([results_dict])

    logging.info("DONE")
    return oie_lines, oie_lines_dict

if __name__ == "__main__":
    # Parse command line arguments
    args = docopt(__doc__)
    inp_fn = args["--in"]
    batch_size = int(args["--batch-size"])
    out_fn = args["--out"]
    cuda_device = int(args["--cuda-device"]) if (args["--cuda-device"] is not None) \
                  else -1
    debug = args["--debug"]

    lines = [line.strip()
            for line in open(inp_fn, encoding = "utf8")]

    oie_lines = run_oie(lines, batch_size, cuda_device, debug)

    # Write to file
    logging.info(f"Writing output to {out_fn}")
    with open(out_fn, "w", encoding = "utf8") as fout:
        fout.write("\n".join(oie_lines))