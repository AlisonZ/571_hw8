import sys
import os
import json
import nltk
import numpy as np
from numpy.linalg import norm

from utils import load_mention_pairs
from FeatureVectorModel import FeatureVectorModel

nltk.download('wordnet_ic')
from nltk.corpus import wordnet as wn, wordnet_ic

DEMONSTRATIVE_WORDS = ['this', 'that', 'these', 'those']
PLURAL_POS = ['NNPS', 'NNS']
PLURAL_PRONOUNS = ['we', 'our', 'they', 'us']

def get_inputs():
    train_pairs = sys.argv[1]
    embedding_file = sys.argv[2]
    test_pairs = sys.argv[3]
    vectors_output = sys.argv[4]
    class_output = sys.argv[5]
    
    if os.path.exists(vectors_output):
        os.remove(vectors_output)
    
    if os.path.exists(class_output):
        os.remove(class_output)
    
    return train_pairs, embedding_file, test_pairs, vectors_output, class_output

def read_embedded_vectors(embedding_file):
    glove_vectors = {}
    with open(embedding_file, 'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            vector = [float(x) for x in split_line[1:]]
            glove_vectors[word] = vector
     
    return glove_vectors

def get_head(mention):
    tokens = mention['tokens']
    token_ids = [token['id'] for token in tokens]
    for token in tokens:
        head_id = token['head_id']
        if head_id not in token_ids:
            return token
    return tokens[-1]

def set_distance(fv, antecedent, anaphor):
    # This makes the assumption that the anaphor will always come after the antecedent
    distance =  anaphor['sentence_num'] - antecedent['sentence_num']
    fv.set_distance(distance)

def is_pronoun(mention):
    tokens = mention['tokens']
    if len(tokens) == 1 and (tokens[0]['pos'] == 'PRP' or tokens[0]['pos'] == 'PRP$'):
        return 1
    else:
        return 0    

def get_clean_string(mention):
    tokens = mention['tokens']
    clean_str_array = []
    # TODO: check for DT? or if the word is in the list of determiners/demonstratives? 
    leading_word_pos = tokens[0]['pos']
    if leading_word_pos == 'DT':
        tokens.pop(0)
    for token in tokens:
        clean_str_array.append(token['word'])
    clean_str = " ".join(clean_str_array)
  
    return clean_str

def set_definite_anaphor(anaphor, fv):
    tokens = anaphor['tokens']
    for token in tokens:
        pos = token['pos']
        word = token['word']
        # TODO: only check for DT and only for "the"??
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
    head_pos = head['pos']
    word = head['word']

    if head_pos in PLURAL_POS:
        return 'plural'
    if (head_pos == 'PRP' or head_pos == 'PRP$') and word in PLURAL_PRONOUNS:
        if word == 'they':
            return 'both'
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

def get_named_entity(head):
    pos = head['pos']
    named_entity = head['named_entity']
    if pos == 'PRP':
        return 'person'
    else:
        return named_entity

def get_embedding_similarity(glove_vectors, antecedent, anaphor):

    ant_tokens = antecedent['tokens']
    anaphor_tokens = anaphor['tokens']

    ant_vector = []
    anaphor_vector = []

  

    ant_average = 0
    for token in ant_tokens:
        word = token['word']
        if word in glove_vectors:
            word_vector = glove_vectors[word]
            ant_vector.append(word_vector)

    for token in anaphor_tokens:
        word = token['word']
        if word in glove_vectors:
            word_vector = glove_vectors[word]
            anaphor_vector.append(word_vector)

    if not ant_vector or not anaphor_vector:
        return 0
    
    ant_average = np.mean(np.array(ant_vector), axis=0)
    anaphor_average = np.mean(np.array(anaphor_vector), axis=0)
    
    cos = np.dot(ant_average, anaphor_average)/(norm(ant_average) * norm(anaphor_average))
    return cos

def print_feature_vectors(fv, vectors_output, label):
    with open(vectors_output, 'a') as output_file:
        print(f'{fv.get_distance()}\t{fv.get_is_antecendent_pronoun()}\t{fv.get_is_anaphor_pronoun()}\t{fv.get_string_match()}\t{fv.get_head_string_match()}\t{fv.get_is_definite_anaphor()}\t{fv.get_is_demonstrative_anaphor()}\t{fv.get_number_agreement()}\t{fv.get_same_named_entity()}\t{fv.get_wordnet_similarity()}\t{fv.get_embedding_similarity()}\t{label}', file=output_file)

def parse_mps(mps, glove_vectors, vectors_output):

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
        if ant_number == 'both' or anaphor_number == 'both':
            fv.set_number_agreement(1)
        else:
            if ant_number == anaphor_number:
                fv.set_number_agreement(1)
            else: 
                fv.set_number_agreement(0)

        # Same named entity
        antecendent_entity = get_named_entity(antecedent_head)
        anaphor_entity = get_named_entity(anaphor_head)
        if antecendent_entity == anaphor_entity:
            fv.set_same_named_entity(1)
        else:
            fv.set_same_named_entity(0)

        # Wordnet similarity
        wordnet_similarity = get_wordnet_similarity(antecedent_head, anaphor_head)

        # Embedding similarity 
        cos = get_embedding_similarity(glove_vectors, antecedent, anaphor)
        fv.set_embedding_similarity(cos)
        label = mp['label']
        print_feature_vectors(fv, vectors_output, label)
    
def main():
    train_pairs, embedding_file, test_pairs, vectors_output, class_output = get_inputs()
    glove_vectors = read_embedded_vectors(embedding_file)

    train_mps = load_mention_pairs(train_pairs)
    parse_mps(mps = train_mps, glove_vectors=glove_vectors, vectors_output = vectors_output)

if __name__ =='__main__':
    main()