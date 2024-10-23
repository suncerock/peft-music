# Parameter-Efficient Fine-Tuning for Music Auto-tagging

Code and results for *Parameter-Efficient Fine-Tuning (PEFT)* for music foundation models.

This branch shows the results with other PEFT methods, on other tasks and datasets.

## Methods
- Adapter (Houlsby et. al., [Parameter-Efficient Transfer Learning for NLP](https://proceedings.mlr.press/v97/houlsby19a/houlsby19a.pdf))
- Prefix Tuning (Li et. al., [Prefix-tuning: Optimizing continuous prompts for generation](https://aclanthology.org/2021.acl-long.353.pdf))
- Bitfit (Zaken et. al., [Bitfit: Simple parameter-efficient fine-tuning for transformer-based masked language-models](https://aclanthology.org/2022.acl-short.1.pdf))
- SSF (Lian et. al., [Scaling & shifting your features: A new baseline for efficient model tuning](https://papers.neurips.cc/paper_files/paper/2022/file/00bb4e415ef117f2dee2fc3b778d806d-Paper-Conference.pdf))
- LoRA (Hu et. al., [LoRA: Low-rank adaptation of large language models](https://openreview.net/forum?id=nZeVKeeFYf9))


## Tasks and Datasets
- Auto tagging
    - MagnaTagATune (MTAT, MTAT-Clean)
    - MTG-Jamendo (MTG-Top50)
- Key detection
    - GiantSteps
- Chord detection
    - Beatles
- Tempo estimation
    - GTZAN
- Beat tracking
    - GTZAN

## Results

| | | MTAT-Clean | MTAT | MTG-Top50 | GiantSteps-Key | Beatles-Chord | GTZAN-Tempo | GTZAN-Beat |
|-|-|------------|------|-----------|----------------|---------------|-------------|-----------|
| | | mAP        | mAP  | mAP       | Weighted Acc.  | Majmin Acc.   | Acc. 1      | F1         |
| **MusicFM**  | **FT**                 | .469     | .393     | .309     | .725     | **.745** | .821     | **.855** |
|              | **FT (reported)**      | .481     | -        | -        | -        | -        | -        | -        |
|              | **Probing**            | .472     | .397     | .297     | .684     | .651     | .817     | .812     |
|              | **Probing (reported)** | .488     | -        | -        | -        | -        | -        | -        |
|              | **Adapter**            | **.491** | **.410** | **.317** | .726     | .736     | .838     | .817     |
|              | **Prefix**             | .487     | .405     | .308     | .729     | .724     | **.855** | .811     |
|              | **Bitfit**             | .479     | .400     | .307     | .702     | .709     | .838     | .794     |
|              | **SSF**                | .481     | .402     | .308     | .709     | .711     | .848     | .796     |
|              | **LoRA**               | .486     | .408     | .316     | **.742** | .726     | .810     | .812     |
| **MERT-95M** | **FT**                 | .448     | .369     | .294     | .722     | .619     | .786     | .877     |
|              | **Probing**            | .468     | .391     | .300     | .672     | .418     | .517     | .870     |
|              | **Probing (reported)** | -        | .393     | .289     | -        | -        | -        | -        |
|              | **Adapter**            | .483     | .405     | .312     | **.734** | .648     | .762     | .887     |
|              | **Prefix**             | .471     | .397     | .300     | .710     | **.649** | .766     | .889     |
|              | **Bitfit**             | .481     | .406     | .310     | .712     | .632     | **.793** | .889     |
|              | **SSF**                | **.484** | .402     | **.313** | .712     | .643     | **.793** | .893     |
|              | **LoRA**               | .481     | **.409** | .311     | .715     | .647     | .786     | **.894** |
| **SOTA**     |                        | .488     | **.414** | **.321** | .731     | .741     | .817     | .871     |


## Usage
Download checkpoints for [MusicFM](https://github.com/minzwon/musicfm) and [MERT](https://huggingface.co/m-a-p/MERT-v1-95M/tree/main) needs to be downloaded first, then update the path to the checkpoints in the config file.

For training, run
```
python train.py PATH/TO/CONFIG.yaml
```