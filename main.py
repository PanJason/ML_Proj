import option
from vertebraLandmark.main import VertebraLandmarkMain

if __name__ == "__main__":
    args = option.read()
    if args.vertebra:
        VertebraLandmarkMain(args)
