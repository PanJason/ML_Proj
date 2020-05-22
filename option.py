import argparse


def read():
    parser = argparse.ArgumentParser(description='Fracture Detection')

    parser.add_argument('--useGPU', action="store_true",
                        default=False, help="Use GPU")

    parser.add_argument('--trainModel', default="None",
                        type=str, help="Specify which model to train")
    parser.add_argument('--continueModel', default="None",
                        type=str, help="Specify which model to continue training")

    parser.add_argument('--data_set', default='data/fracture/train_processed',
                        type=str, help='Path to the dataset')
    parser.add_argument('--anno_path', default='data/fracture/annotations/anno_train.json',
                        type=str, help='Path to the dataset annotation')
    parser.add_argument('--output_path', default='result/result.json',
                        type=str, help='Path to the output file')
    parser.add_argument('--model_path', default='saved_model',
                        type=str, help='Path to the saved model')
    parser.add_argument('--addi_path', default='additional_anno/additional_anno_train.json',
                        type=str, help='Path to the additional dataset annotation')
    parser.add_argument('--val_data_set', default='data/fracture/val_processed',
                        type=str, help='Path to the val dataset')
    parser.add_argument('--val_addi_path', default='additional_anno/additional_anno_val.json',
                        type=str, help='Path to the additional val dataset annotation')

    parser.add_argument('--batchSize', default=16, type=int,
                        help='Batch size')
    parser.add_argument('--numEpochs', default=10, type=int,
                        help='Number of epochs to run')
    parser.add_argument('--learningRate', default=1e-3, type=float,
                        help='Learning rate')

    parser.add_argument('--regionSize', default=200, type=int,
                        help='Region size')
    parser.add_argument('--imageSize', default=3056, type=int,
                        help='Region size')
    parser.add_argument('--tracerSampleRate', default=0.02, type=float,
                        help='Samples per pixel use in tracer')
    parser.add_argument('--regionShiftSigma', default=20.0, type=float,
                        help='Sigma for gaussians used in region shifting')
    parser.add_argument('--maxTrace', default=10, type=int,
                        help="Maximum number of regions selected by a single trace")
    parser.add_argument('--traceStep', default=0.5, type=float,
                        help="Step of each trace sample")
    parser.add_argument('--actorLearningRate', default=1e-5, type=float,
                        help='Actor Learning rate')
    parser.add_argument('--criticLearningRate', default=1e-4, type=float,
                        help='Critic Learning rate')

    parser.add_argument('--maxSteps', default=10, type=int,
                        help="Max step inside a DDPG run")
    parser.add_argument('--DDPGHiddenSize', default=100, type=int,
                        help="Hidden size inside rnn of DDPG")
    parser.add_argument('--DDPGtau', default=0.001, type=float,
                        help="Tau coefficient in DDPG")
    parser.add_argument('--DDPGgamma', default=0.99, type=float,
                        help="Gamma coefficient in DDPG")
    parser.add_argument('--DDPGepsilonDec', default=4e-5, type=float,
                        help="Decrease of epsilon coeficcient in DDPG")
    parser.add_argument('--failReward', default=-1000, type=float,
                        help='The reward when failure meets')
    parser.add_argument('--finishReward', default=1000, type=float,
                        help='The reward when finisha trace')
    parser.add_argument('--rewardScale', default=0.001, type=float,
                        help='The scale of reward')
    parser.add_argument('--warmUpSize', default=2000, type=int,
                        help="Records needed for warm up in DDPG")
    parser.add_argument('--maxBuffer', default=10000, type=int,
                        help="Max buffer size in DDPG")
    parser.add_argument('--OUtheta', default=0.8, type=float,
                        help="Theta coefficient in OU")
    parser.add_argument('--OUsigma', default=0.3, type=float,
                        help="Sigma coefficient in OU")
    parser.add_argument('--OUdt', default=1e-2, type=float,
                        help="dt coefficient in OU")

    parsed = parser.parse_args()

    print("Fracture detection")
    print("Args:")
    for u, v in parsed.__dict__.items():
        print(f"{u} : {v}")

    return parsed
