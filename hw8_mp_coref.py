import sys
from utils import load_mention_pairs
from FeatureVectorModel import FeatureVectorModel
import json
import nltk
nltk.download('wordnet_ic')
from nltk.corpus import wordnet as wn, wordnet_ic

DEMONSTRATIVE_WORDS = ['this', 'that', 'these', 'those']
PLURAL_POS = ['NNPS', 'NNS']
PLURAL_PRONOUNS = ['we', 'our', 'they', 'us']

def get_inputs():
    train_pairs = sys.argv[1]
    test_pairs = sys.argv[2]
    vectors_output = sys.argv[3]
    class_output = sys.argv[4]
    return train_pairs, test_pairs, vectors_output, class_output

def read_embedded_vectors():
    print("Vectors")
    # /mnt/dropbox/25-26/571W/hw8/dolma_300_2024_1.2M.100_combined.txt

def get_head(mention):
    tokens = mention['tokens']
    token_ids = [token['id'] for token in tokens]
    for token in tokens:
        head_id = token['head_id']
        if head_id not in token_ids:
            return token
        else:
            return tokens[-1]

def set_distance(fv, antecedent, anaphor):
    # TODO: is the correc way to calculate this? Do we need to check which comes first?
    # TODO: if both mentions appear in the sames sentence, value should be 0 --> what does an example like this looks like? How to tell?
    distance =  antecedent['sentence_num'] - anaphor['sentence_num']
    if distance > 0:
        fv.set_distance(distance)
    else:
        switched_distance = anaphor['sentence_num'] - antecedent['sentence_num'] 
        fv.set_distance(switched_distance)

def is_pronoun(mention):
    tokens = mention['tokens']
    if len(tokens) == 1 and (tokens == 'PRP' or tokens == '$PRP'):
        return 1
    else:
        return 0    

# TODO: only check for DT? anything else? return long strings of all tokens?
def get_clean_string(mention):
    tokens = mention['tokens']
    clean_str_array = []
    for token in tokens:
        if token['pos'] != 'DT':
            clean_str_array.append(token['word'])
    clean_str = " ".join(clean_str_array)
  
    return clean_str

# TODO: only check for DT and only for "the"??
# Is this handling this correctly?
def set_definite_anaphor(anaphor, fv):
    tokens = anaphor['tokens']
    for token in tokens:
        pos = token['pos']
        word = token['word']
        if pos == 'DT' and word == 'the':
            fv.set_is_definite_anaphor(1)
            return
    fv.set_is_definite_anaphor(0)
    return

def set_demonstrative_anaphor(anaphor, fv):
    tokens = anaphor['tokens']
    for token in tokens:
        pos = token['pos']
        word = token['word']
        if pos == 'DT' and word in DEMONSTRATIVE_WORDS:
            fv.set_is_demonstrative_anaphor(1)
            return
    fv.set_is_demonstrative_anaphor(0)
    return

def get_number(head):
    # TODO: check this logic and grammar rules for returning plural and singular
    head_pos = head['pos']
    word = head['word']

    if head_pos in PLURAL_POS:
        return 'plural'
    if (head_pos == 'PRP' or head_pos == '$PRP') and word in PLURAL_PRONOUNS:
        return 'plural'
    else:
        return 'singular' 

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('NN'): return wn.NOUN
    if pos_tag.startswith('VB'): return wn.VERB
    if pos_tag.startswith('JJ'): return wn.ADJ
    if pos_tag.startswith('RB'): return wn.ADV
    return None

def get_wordnet_similarity(antecedent_head, anaphor_head):
    brown_ic = wordnet_ic.ic('ic-brown.dat')
    ant_pos = get_wordnet_pos(antecedent_head['pos'])
    anaphor_pos = get_wordnet_pos(anaphor_head['pos'])
    synset_1 = wn.synsets(antecedent_head['lemma'], pos=ant_pos)
    synset_2 = wn.synsets(anaphor_head['lemma'], pos=anaphor_pos)

    if not synset_1 or not synset_2:
        return -1

    max_sim = -1
    for synset1 in synset_1:
        for synset2 in synset_2:
            if synset1._pos != synset2._pos:
                continue
            sim_score = synset1.res_similarity(synset2, brown_ic)
            if sim_score > max_sim:
                max_sim = sim_score
    return max_sim

# TODO: Implement with the previous assignment
def get_embedding_similarity():
    # Float:  Compute the cosine similarity of the two full mention phrases using the static Glove vectors from the previous assignment. 
    # NOTE: If none of the words in one (or more) the mention phrases are in the embedding file, return 0 for the similarity.

def parse_mps(mps):
    for mp in mps:
        fv = FeatureVectorModel()
        antecedent = mp['antecedent']
        anaphor = mp['anaphor']
        antecedent_head = get_head(mention=antecedent)
        anaphor_head = get_head(mention=anaphor)

        # Distance
        set_distance(fv,antecedent, anaphor)
        
        # Antecedent Pronoun
        antecedent_is_pronoun = is_pronoun(mention = antecedent)
        fv.set_is_antecendent_pronoun(antecedent_is_pronoun)
        
        # Anaphor Pronoun
        anaphor_is_pronoun = is_pronoun(mention = anaphor)
        fv.set_is_anaphor_pronoun(anaphor_is_pronoun)
        
        # String match
        ant_string = get_clean_string(mention = antecedent)
        anaphor_string = get_clean_string(mention = anaphor)
        if ant_string == anaphor_string:
            fv.set_string_match(1)
        else:
            fv.set_string_match(0)
        
        # Head string match
        if antecedent_head['word'] == anaphor_head['word']:
            fv.set_head_string_match(1)
        else:
            fv.set_head_string_match(0)

        # Definite anaphor
        set_definite_anaphor(anaphor,fv)

        # Demonstrative anaphor
        set_demonstrative_anaphor(anaphor, fv)

        # Number agreement
        ant_number = get_number(head=antecedent_head)
        anaphor_number = get_number(head=anaphor_head)
        if ant_number == anaphor_number:
            fv.set_number_agreement(1)
        else: 
            fv.set_number_agreement(0)

        # Same named entity
        # TODO: is this the only thing to check with named entity? all are __
        antecendent_entity = antecedent_head['named_entity']
        anaphor_entity = anaphor_head['named_entity']
        if antecendent_entity == anaphor_entity:
            fv.set_same_named_entity(1)
        else:
            fv.set_same_named_entity(0)

        # Wordnet similarity
        wordnet_similarity = get_wordnet_similarity(antecedent_head, anaphor_head)

        # Embedding similarity 
        # TODO: Implement and call
        # get_embedding_similarity()
        

    
def main():
    train_pairs, test_pairs, vectors_output, class_output = get_inputs()
    # read_embedded_vectors()
    train_mps = load_mention_pairs(train_pairs)
    parse_mps(mps = train_mps)

if __name__ =='__main__':
    main()