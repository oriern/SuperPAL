# SuperPAL

Data, Code and Model for the paper "[SuperPAL: Supervised Proposition ALignment for Multi-Document Summarization and Derivative Sub-Tasks](https://arxiv.org/abs/2009.00590)".

You can try [SuperPAL aligner demo](https://nlp.biu.ac.il/~ernstor1/SuperPAL_IU/) for a sense.


## Data generation ##

Predicted alignments of MultiNews and CNN/DailyMail train and val datasets can be found [here](https://drive.google.com/drive/folders/1JnRrdbENzBLpbae5ZIKmil1fuZhm2toc?usp=sharing).

To generate derived datasets (salience, clustering and generation) out of an alignment file use:
```
  python createSubDatasets.py -alignments_path <ALIGNMENTS_PATH>  -out_dir_path <OUT_DIR_PATH>
```

`transformers` directory was forked from [huggingface](https://github.com/huggingface/transformers) v2.5.1, and edited for our purpose.
`supervised_oie_wrapper` directory is a wrapper over AllenNLP's pretrained Open IE model that was implemented by Gabriel Stanovsky. It was forked from [here](https://github.com/gabrielStanovsky/supervised_oie_wrapper), and edited for our purpose.

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
