""" Usage:
    <file-name> --in=INPUT_FILE --out=OUTPUT_FILE [--debug]
"""
# External imports
import logging
from pprint import pprint
from pprint import pformat
from docopt import docopt
import json
from collections import defaultdict
from tqdm import tqdm
from allennlp.pretrained import open_information_extraction_stanovsky_2018
from allennlp.predictors.open_information_extraction import consolidate_predictions
from allennlp.predictors.open_information_extraction import join_mwp
from allennlp.predictors.open_information_extraction import make_oie_string
from allennlp.predictors.open_information_extraction import get_predicate_text

# Local imports

#=----

class Mock_token:
    """
    Spacy token imitation
    """
    def __init__(self, tok_str):
        self.text = tok_str

    def __str__(self):
        return self.text

def get_oie_frame(tokens, tags) -> str:
    """
    Converts a list of model outputs (i.e., a list of lists of bio tags, each
    pertaining to a single word), returns an inline bracket representation of
    the prediction.
    """
    frame = defaultdict(list)
    chunk = []
    words = [token.text for token in tokens]

    for (token, tag) in zip(words, tags):
        if tag.startswith("I-") or tag.startswith("B-"):
            frame[tag[2:]].append(token)

    return dict(frame)


def get_frame_str(oie_frame) -> str:
    """
    Convert and oie frame dictionary to string.
    """
    dummy_dict = dict([(k if k != "V" else "ARG01", v)
                       for (k, v) in oie_frame.items()])

    sorted_roles = sorted(dummy_dict)

    frame_str = []
    for role in sorted_roles:
        if role == "ARG01":
            role = "V"
        arg = " ".join(oie_frame[role])
        frame_str.append(f"{role}:{arg}")

    return "\t".join(frame_str)


def format_extractions(sent_tokens, sent_predictions):
    """
    Convert token-level raw predictions to clean extractions.
    """
    # Consolidate predictions
    if not (len(set(map(len, sent_predictions))) == 1):
        raise AssertionError
    assert len(sent_tokens) == len(sent_predictions[0])
    sent_str = " ".join(map(str, sent_tokens))

    pred_dict = consolidate_predictions(sent_predictions, sent_tokens)

    # Build and return output dictionary
    results = []
    all_tags = []
    results_dict = {'verbs':[], 'words': [str(token) for token in sent_tokens]}

    for tags in pred_dict.values():
        # Join multi-word predicates
        tags = join_mwp(tags)
        all_tags.append(tags)

        # Create description text
        oie_frame = get_oie_frame(sent_tokens, tags)

        # Add a predicate prediction to outputs.
        results.append("\t".join([sent_str, get_frame_str(oie_frame)]))
        results_dict['verbs'].append({'tags': tags})

    return results, all_tags, results_dict