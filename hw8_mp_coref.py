import sys

def get_inputs():
    train_pairs = sys.argv[1]
    test_pairs = sys.argv[2]
    vectors_output = sys.argv[3]
    class_output = sys.argv[4]
    return train_pairs, test_pairs, vectors_output, class_output

def main():
    train_pairs, test_pairs, vectors_output, class_output = get_inputs()

if __name__ =='__main__':
    main()