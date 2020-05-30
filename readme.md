# ML project: fracture detection

This is a project for the course Machine Learning. The purpose of this project is to detect fractures in chest bones.

## Requirement

torch 1.5.0+cu101
torchvision 0.6.0+cu101
numpy 1.18.4
pandas 1.0.3
Pillow 7.1.2
seaborn 0.10.1
Cython 0.29.19
matplotlib 3.1.3
scipy 1.4.1
pycocotools 2.0
opencv

If running in Windows, pycocotools is incompatible with numpy 1.18.4, please downgrade numpy to 1.17.4.

## Usage

```bash
python test.py --data_dir /path/to/fracture/test \
               --anno_dir /path/to/anno_val.json \
               --output_path /path/to/output/result.json \
               --model_path saved_model
```

The trained model is contained in ./saved_model.
