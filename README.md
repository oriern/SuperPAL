# SuperPAL

Data, Code and Model for the paper "[Summary-Source Proposition-level Alignment: Task, Datasets and Supervised Baseline](https://aclanthology.org/2021.conll-1.25.pdf)".

You can use our huggingface model or check our demo [here](https://huggingface.co/biu-nlp/superpal).


`transformers` directory was forked from [huggingface](https://github.com/huggingface/transformers) v2.5.1, and edited for our purpose.

`supervised_oie_wrapper` directory is a wrapper over AllenNLP's pretrained Open IE model that was implemented by Gabriel Stanovsky. It was forked from [here](https://github.com/gabrielStanovsky/supervised_oie_wrapper), and edited for our purpose.

## Manual Datasets ##

All manual datasets are under `manual_datasets` repository, including crowdsourced dev and test sets, and Pyramid-based train set.

As DUC-based datasets are limited to LDC agreement, we provide here only the character index of all propositions or sentences.

So, if you have the original dataset, you can regenerate the alignments easily.

If you have any issue regarding the DUC alignment regeneration, please contact via email.

In addition, we are trying to upload our alignment datasets to LDC, so it will not have agreement issues. Will be updated soon.


MultiNews alignments are released in full.



## Data generation ##

Predicted alignments of MultiNews and CNN/DailyMail train and val datasets can be found [here](https://drive.google.com/drive/folders/1JnRrdbENzBLpbae5ZIKmil1fuZhm2toc?usp=sharing).

To generate derived datasets (salience, clustering and generation) out of an alignment file use:
```
  python createSubDatasets.py -alignments_path <ALIGNMENTS_PATH>  -out_dir_path <OUT_DIR_PATH>
```

## Alignment model ##
To apply aligment model on your own data, follow the following steps:
  1. download the trained model [here](https://drive.google.com/drive/folders/1kTaZQVxUm-RWbF71QpOue5xDuV7-IP2i?usp=sharing) and put it under       `/transformers/examples/out/outnewMRPC_OIU/SpansOieNegativeAll_pan_full089_fixed/checkpoint-2000/`

  2. run
  ```
  python main_predict.py -data_path <DATA_PATH>  -output_path <OUT_DIR_PATH>  -alignment_model_path  <ALIGNMENT_MODEL_PATH>
  ```
  `<DATA_PATH>` should contain the following structure where a summary and its related document directory share the same name:
      
      - <DATA_PATH>
        - summaries
          - A.txt
          - B.txt
          - ...
        - A
          - doc_A1
          - doc_A2
          - ...
        - B
          - doc_B1
          - doc_B2
          - ...
         
  3. It will create two files in `<OUT_DIR_PATH>`:
    - 'dev.tsv' - contains all alignment candidate pairs.
    - a '.csv' file - contains all predicted aligned pairs with their classification score.
