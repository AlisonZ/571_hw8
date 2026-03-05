class FeatureVectorModel:
    def __init__(self):
        self.distance = 0
        self.is_antecendent_pronoun = False, 
        self.is_anaphor_pronoun = False, 
        self.string_match = 0
        self.head_string_match = 0
        self.is_definite_anaphor = 0
        self.is_demonstrative_anaphor = 0
        self.number_agreement = 0
        self.same_named_entity = 0
        self.wordnet_similarity = 0
        self.embedding_similarity = 0 

    def set_distance(self, distance):
        self.distance = distance

    def set_is_antecendent_pronoun(self, is_antecendent_pronoun):
        self.is_antecendent_pronoun = is_antecendent_pronoun

    def set_is_anaphor_pronoun(self, is_anaphor_pronoun):
        self.is_anaphor_pronoun = is_anaphor_pronoun

    def set_string_match(self, string_match):
        self.string_match = string_match
    
    def set_head_string_match(self, head_string_match):
        self.head_string_match = head_string_match

    def set_is_definite_anaphor(self, is_definite_anaphor):
        self.is_definite_anaphor = is_definite_anaphor

    def set_is_demonstrative_anaphor(self, is_demonstrative_anaphor):
        self.is_demonstrative_anaphor = is_demonstrative_anaphor

    def set_number_agreement(self, number_agreement):
        self.number_agreement = number_agreement

    def set_same_named_entity(self, same_named_entity):
        self.same_named_entity = same_named_entity

    def set_wordnet_similarity(self, wordnet_similarity):
        self.wordnet_similarity = wordnet_similarity

    def set_embedding_similarity(self, embedding_similarity):
        self.embedding_similarity = embedding_similarity
    
    def get_distance(self):
        return self.distance
    
    def get_is_antecendent_pronoun(self):
        return self.is_antecendent_pronoun

    def get_is_anaphor_pronoun(self):
        return self.is_anaphor_pronoun

    def get_string_match(self):
        return self.string_match

    def get_head_string_match(self):
        return self.head_string_match

    def get_is_definite_anaphor(self):
        return self.is_definite_anaphor

    def get_is_demonstrative_anaphor(self):
        return self.is_demonstrative_anaphor

    def get_number_agreement(self):
        return self.number_agreement

    def get_same_named_entity(self):
        return self.same_named_entity

    def get_wordnet_similarity(self):
        return self.wordnet_similarity

    def get_embedding_similarity(self):
        return self.embedding_similarity