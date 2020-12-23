# SuperPAL

Data, Code and Model for the paper "[SuperPAL: Supervised Proposition ALignment for Multi-Document Summarization and Derivative Sub-Tasks](https://arxiv.org/abs/2009.00590)".

You can try [SuperPAL aligner demo](https://nlp.biu.ac.il/~ernstor1/SuperPAL_IU/) for a sense.

Predicted alignments of MultiNews and CNN/DailyMail train and val datasets can be found [here](https://drive.google.com/drive/folders/1JnRrdbENzBLpbae5ZIKmil1fuZhm2toc?usp=sharing).

To generate derived datasets (salience, clustering and generation) out of an alignment file use:
```
  python createSubDatasets.py -alignments_path <ALIGNMENTS_PATH>  -out_dir_path <OUT_DIR_PATH>
```

## Alignment model ##
To apply aligment model on your own data, follow the following steps:
  1. download the trained model [here](https://drive.google.com/drive/folders/1kTaZQVxUm-RWbF71QpOue5xDuV7-IP2i?usp=sharing) and put it under       '/transformers/examples/out/outnewMRPC_OIU/SpansOieNegativeAll_pan_full089_fixed/checkpoint-2000/'

  2. run
```
  python main_predict.py -data_path <DATA_PATH>  -output_path <OUT_DIR_PATH>  -alignment_model_path  <ALIGNMENT_MODEL_PATH>
```
It will create two files in <OUT_DIR_PATH>:
  - 'dev.tsv' - contains all alignment candidate pairs.
  - a '.csv' file - contains predicted aligned pairs with their classification score.
