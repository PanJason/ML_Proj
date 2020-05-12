import argparse


def read():
    parser = argparse.ArgumentParser(description='Fracture Detection')

    parser.add_argument('--useGPU', action="store_true",
                        default=False, help="Use GPU")

    parser.add_argument('--trainModel', default="None",
                        type=str, help="Specify which model to train")
    parser.add_argument('--continueModel', default="None",
                        type=str, help="Specify which model to continue training")

    parser.add_argument('--data_set', default='data/fracture/train',
                        type=str, help='Path to the dataset')
    parser.add_argument('--anno_path', default='data/fracture/annotations/anno_train.json',
                        type=str, help='Path to the dataset annotation')
    parser.add_argument('--output_path', default='result/result.json',
                        type=str, help='Path to the output file')
    parser.add_argument('--model_path', default='saved_model',
                        type=str, help='Path to the saved model')
    parser.add_argument('--addi_path', default='additional_anno/additional_anno_train.json',
                        type=str, help='Path to the additional dataset annotation')
    parser.add_argument('--val_data_set', default='data/fracture/val',
                        type=str, help='Path to the val dataset')
    parser.add_argument('--val_addi_path', default='additional_anno/additional_anno_val.json',
                        type=str, help='Path to the additional val dataset annotation')

    parser.add_argument('--batchSize', default=4, type=int,
                        help='Batch size')
    parser.add_argument('--numEpochs', default=10, type=int,
                        help='Number of epochs to run')
    parser.add_argument('--learningRate', default=1e-3, type=float,
                        help='Learning rate')

    parser.add_argument('--regionSize', default=200, type=int,
                        help='Region size')
    parser.add_argument('--tracerSampleRate', default=0.05, type=float,
                        help='Samples per pixel use in tracer')
    parser.add_argument('--regionShiftSigma', default=20.0, type=float,
                        help='Sigma for gaussians used in region shifting')

    parsed = parser.parse_args()

    print("Fracture detection")
    print("Args:")
    for u, v in parsed.__dict__.items():
        print(f"{u} : {v}")

    return parsed
