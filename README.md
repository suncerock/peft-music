# Parameter-Efficient Fine-Tuning for Music Auto-tagging

Code and results for *Parameter-Efficient Fine-Tuning (PEFT)* for music foundation models.

The main branch serves as supplementary material for the paper and therefore contains only the results in the paper. For extended results with other methods and results on other tasks, please check the other branch.

## Methods
- Adapter (Houlsby et. al., [Parameter-Efficient Transfer Learning for NLP](https://proceedings.mlr.press/v97/houlsby19a/houlsby19a.pdf))
- LoRA (Hu et. al., [LoRA: Low-rank adaptation of large language models](https://openreview.net/forum?id=nZeVKeeFYf9))

## Tasks and Datasets
- Auto tagging
    - MagnaTagATune (MTAT, MTAT-Clean)
    - MTG-Jamendo (MTG-Top50)

## Results

| | | MTAT-Clean |                | MTAT |                | MTG-Top50 |         |
|-|-|------------|----------------|------|----------------|-----------|---------|
| | | mAP        | ROC-AUC        | mAP  | ROC-AUC        | mAP       | ROC-AUC |
| **MusicFM**  | **FT**                 | .469     | .918     | .393     | .911     | .309     | .837     |
|              | **FT (reported)**      | .481     | .920     | -        | -        | -        | -        |
|              | **Probing**            | .472     | .917     | .397     | .911     | .297     | .833     |
|              | **Probing (reported)** | .488     | **.924** | -        | -        | -        | -        |
|              | **Adapter**            | **.491** | **.924** | **.410** | **.917** | **.317** | **.840** |
|              | **LoRA**               | .486     | **.924** | .408     | **.917** | .316     | **.840** |
| **MERT-95M** | **FT**                 | .448     | .910     | .369     | .901     | .294     | .828     |
|              | **Probing**            | .468     | .917     | .391     | .910     | .300     | .828     |
|              | **Probing (reported)** | -        | -        | .393     | .910     | .289     | .830     |
|              | **Adapter**            | .483     | .922     | .405     | .915     | .312     | **.840** |
|              | **LoRA**               | .482     | .922     | .407     | .916     | .311     | **.840** |
| **SOTA**     |                        | .488     | **.924** | **.414** | **.927** | **.321** | **.843** |


## Usage
Download checkpoints for [MusicFM](https://github.com/minzwon/musicfm) and [MERT](https://huggingface.co/m-a-p/MERT-v1-95M/tree/main) needs to be downloaded first, then update the path to the checkpoints in the config file.

For training, run
```
python train.py PATH/TO/CONFIG.yaml
```