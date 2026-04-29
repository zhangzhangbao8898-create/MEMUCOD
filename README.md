# MEMUCOD

## Environment

Enter the project directory:

```bat
cd /d E:\MEMUCOD
```

Create the conda environment:

```bat
conda env create -f environment.yml
```

Activate the environment:

```bat
conda activate memucod
```

Verify PyTorch and CUDA:

```bat
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## Download

The training data and model checkpoint are available from Google Drive:

```text
https://drive.google.com/drive/folders/1PCKbsU9FZjO7vZNh3Sa19fHb8Je6vFax?usp=drive_link
```

## Training

Prepare the training data with this layout:

```text
MEMUCOD/
  data/
    train/
      COD600/
        Imgs/
        GT/
      SOD600/
        Imgs/
        GT/
```

Run training from the `src` directory:

```bat
cd src
python train.py
```

## Testing

Prepare the checkpoint and test data with this layout:

```text
MEMUCOD/
  checkpoints/
    memucod/
      MEMUCOD_10.pth
  data/
    test/
      CHAMELEON/
        Imgs/
      CAMOtest250/
        Imgs/
```

Run inference from the `src` directory:

```bat
cd src
python test.py
```

Predictions are saved to:

```text
MEMUCOD/outputs/memucod/
```

## Evaluation

Check that prediction and ground-truth paths in the evaluation JSON files are valid:

```bat
python PySODEvalToolkit/tools/check_path.py --method-jsons eval/method_output.json --dataset-jsons eval/dataset_GT.json
```

Run evaluation with the default JSON files:

```bat
python PySODEvalToolkit/eval.py --dataset-json eval/dataset_GT.json --method-json eval/method_output.json --metric-names msiou bioa fmeasure wfm sm em --num-workers 4 --record-txt output/binary_results.txt
```

Run evaluation with common segmentation metrics:

```bat
python PySODEvalToolkit/eval.py --dataset-json eval/dataset_GT.json --method-json eval/method_output.json --metric-names sm wfm mae iou dice f1 em --num-workers 4 --record-txt output/binary_results.txt
```

CAMO example:

```bat
python PySODEvalToolkit/eval.py --dataset-json eval/camoGT.json --method-json eval/camopre.json --metric-names sm wfm mae iou dice f1 em --num-workers 16 --record-txt output/binary_results.txt
```

Test-set example:

```bat
python PySODEvalToolkit/eval.py --dataset-json eval/testgt.json --method-json eval/testpre.json --metric-names sm wfm mae iou dice f1 em --num-workers 16 --record-txt output/binary_results.txt
```

Optional output formats:

```bat
python eval.py ^
  --dataset-json dataset_binary.json ^
  --method-json method_binary.json ^
  --metric-names miou bioa fmeasure fm wfm sm em em_mean ^
  --num-workers 4 ^
  --record-txt output/binary_results.txt ^
  --record-xlsx output/binary_results.xlsx
```

## Evaluation JSON Example

Example dataset entry:

```json
{
  "NC4K": {
    "mask": {
      "path": "E:\\MEMUCOD\\data\\CAMO-V.1.0-CVIU2019\\GT",
      "prefix": "camourflage_",
      "suffix": ".png"
    }
  }
}
```
