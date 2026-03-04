# Utility script to load mention-pair data for Ling571 HW#8
import json

def load_mention_pairs(pairfilename):
    """
    load_mention_pairs:
    Argument: pairfilename
        the name of the file with the mention-pair data in JSON format.
    Returns: Array of mention-pair data as specified below

    The data itself is in the following form:
    A python list of pairs of mentions and the pair's label, the pairs are
    dictionaries of the form:
 
    {'antecedent_id': identifier for the mention which is the proposed antecedent,
    'antecedent': Python dictionary with all information about the mention, which 
                  can be used to extract features for classification.
                  Details below.
    'anaphor_id': identifier for the mention which is the proposed anaphor,
    'anaphor': Python dictionary of same structure as 'antecedent' mention
    'label': 1 if the mentions *DO* corefer, 0 otherwise
    }


    A mention has the following form:
    { 'docname': Name of the document in which the mention appeared (str),
    'sentence_num': Index of the sentence in which mention appeared (int),
    'start_token': Index of the starting word of the mention in the sentence (int),
    'end_token': Index of the last word of the mention in that sentence (int)
    'entity_id': All mentions which corefer will have the same entity ID, 
               *Do not use this information for classification*, though
               you may use it in later analysis if you wish
    'token': List of all the tokens in the mention text, stored as a dict,
           Includes additional syntactic and semantic information, as below.
    }

    Tokens include the following information as a dictionary.  All syntactic
    and semantic values were manually created.
    { 'id': Index of word in original sentence,
    'word': word itself,
    'lemma': root form of the word, usually without inflection, often lowercase,
    'pos': part-of-speech tag, in Penn Treebank form,
    'head_id': Index for the dependency head of this word in a dependency parse,
    'deprel': Dependency relation between this word and its head,
    'named_entity': Named Entity tag for this word, e.g. person, norg,
                  or '_', if not a named entity
    }
 
"""

    with open(pairfilename,'r') as pair_f:
        mention_pairs =  [json.loads(x) for x in pair_f]

    return(mention_pairs)

