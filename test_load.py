# Mini script using load_mention_pairs and accessing a word in antecedent
# Usage: $0 <mention_pair_json_filename>, e.g. train_pairs.json

from utils import load_mention_pairs

import sys
def main():
    infile = sys.argv[1]
    mps = load_mention_pairs(infile)
    print(mps[0]['antecedent']['tokens'][0]['word'])
main()
