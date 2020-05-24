import argparse


def read():
    parser = argparse.ArgumentParser(description='Fracture Detection')

    parser.add_argument('--useGPU', action="store_true",
                        default=False, help="Use GPU")

    parser.add_argument('--trainModel', default="None",
                        type=str, help="Specify which model to train")
    parser.add_argument('--continueModel', default="None",
                        type=str, help="Specify which model to continue training")
    parser.add_argument('--vertebra',  default=False, action="store_true",
                        help="Call vertebra landmark's main function")
    parser.add_argument('--testTarget', default="All", type=str,
                        help="Target of test")
    parser.add_argument('--median', default='median', type=str,
                        help="Path to median result files")

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

    parser.add_argument('--batchSize', default=256, type=int,
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
    parser.add_argument('--maxTrace', default=20, type=int,
                        help="Maximum number of regions selected by a single trace")
    parser.add_argument('--traceStep', default=0.8, type=float,
                        help="Step of each trace sample")
    parser.add_argument('--actorLearningRate', default=1e-4, type=float,
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
    parser.add_argument('--DDPGepsilonDelta', default=1e-3, type=float,
                        help="Decrease of epsilon coeficcient in DDPG")
    parser.add_argument('--failReward', default=-10000, type=float,
                        help='The reward when failure meets')
    parser.add_argument('--finishReward', default=1000, type=float,
                        help='The reward when finisha trace')
    parser.add_argument('--rewardScale', default=0.01, type=float,
                        help='The scale of reward')
    parser.add_argument('--warmUpSize', default=1000, type=int,
                        help="Records needed for warm up in DDPG")
    parser.add_argument('--maxBuffer', default=10000, type=int,
                        help="Max buffer size in DDPG")
    parser.add_argument('--updateTimes', default=3, type=int,
                        help="Updates each time")
    parser.add_argument('--OUtheta', default=0.8, type=float,
                        help="Theta coefficient in OU")
    parser.add_argument('--OUsigma', default=0.3, type=float,
                        help="Sigma coefficient in OU")
    parser.add_argument('--OUdt', default=1e-1, type=float,
                        help="dt coefficient in OU")

    parser.add_argument('--num_epoch', type=int,
                        default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int,
                        default=2, help='Number of epochs')
    parser.add_argument('--num_workers', type=int,
                        default=4, help='Number of workers')
    parser.add_argument('--init_lr', type=float,
                        default=1.25e-4, help='Init learning rate')
    parser.add_argument('--down_ratio', type=int, default=4, help='down ratio')
    parser.add_argument('--input_h', type=int,
                        default=1024, help='input height')
    parser.add_argument('--input_w', type=int, default=512, help='input width')
    parser.add_argument('--K', type=int, default=100,
                        help='maximum of objects')
    parser.add_argument('--conf_thresh', type=float,
                        default=0.2, help='confidence threshold')
    parser.add_argument('--seg_thresh', type=float,
                        default=0.5, help='confidence threshold')
    parser.add_argument('--num_classes', type=int,
                        default=1, help='number of classes')
    parser.add_argument('--ngpus', type=int, default=0, help='number of gpus')
    parser.add_argument('--resume', type=str,
                        default='model_last.pth', help='weights to be resumed')
    parser.add_argument('--data_dir', type=str,
                        default='../../Datasets/spinal/', help='data directory')
    parser.add_argument('--phase', type=str,
                        default='test', help='data directory')

    parsed = parser.parse_args()

    print("Fracture detection")
    print("Args:")
    for u, v in parsed.__dict__.items():
        print(f"{u} : {v}")

    return parsed
