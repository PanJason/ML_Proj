import argparse


def read():
    parser = argparse.ArgumentParser(description='Fracture Detection')

    parser.add_argument('-dataset', default='data/fracture',
                        type=str, help='Path to the dataset')

    parser.add_argument('-batchSize', default=8, type=int,
                        help='Batch size')
    parser.add_argument('-numEpochs', default=100, type=int,
                        help='Number of epochs to run')
    parser.add_argument('-learningRate', default=1e-3, type=float,
                        help='Learning rate')

    parsed =parser.parse_args()

    print("Fracture detection")
    print("Args:")
    for u, v in parsed.__dict__.items():
        print(f"{u} : {v}")

    return parsed
