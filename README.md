# SuperPAL

Data, Code and Model for the paper "[Summary-Source Proposition-level Alignment: Task, Datasets and Supervised Baseline](https://aclanthology.org/2021.conll-1.25.pdf)".

 If you find the code useful, please cite the following paper.
  ```
  @inproceedings{ernst-etal-2021-summary,
    title = "Summary-Source Proposition-level Alignment: Task, Datasets and Supervised Baseline",
    author = "Ernst, Ori  and Shapira, Ori  and Pasunuru, Ramakanth  and Lepioshkin, Michael  and Goldberger, Jacob  and Bansal, Mohit  and Dagan, Ido", booktitle = "Proceedings of the 25th Conference on Computational Natural Language Learning", month = nov, year = "2021", publisher = "Association for Computational Linguistics", url = "https://aclanthology.org/2021.conll-1.25", pages = "310--322",}
  ```

You can use our huggingface model or check our demo [here](https://huggingface.co/biu-nlp/superpal).


`run_glue.py` script was forked from [huggingface](https://github.com/huggingface/transformers) v2.5.1, and edited for our purpose.

`supervised_oie_wrapper` directory is a wrapper over AllenNLP's (v0.9.0) pretrained Open IE model that was implemented by Gabriel Stanovsky. It was forked from [here](https://github.com/gabrielStanovsky/supervised_oie_wrapper), and edited for our purpose.

In this repository we used python-3.6. Please refer to `environment_superPAL.yml` for other requirements.


## Manual Datasets ##

All manual datasets are under `manual_datasets` repository, including crowdsourced dev and test sets, and Pyramid-based train set.

As DUC-based datasets are limited to LDC agreement, we provide here only the character index of all propositions or sentences.

To restore the text alignments please use:
```
  python manual_datasets/restore_alignments.py -indx_csv_path <PATH_TO_THE_CSV_WITH_ALIGNMENTS_INDEXES>  -documents_path <PATH_TO_THE_DOCUMENTS_ARANGED_BY_TOPIC_DIRECTORIES> -summaries_path <SUMMARIES_PATH> -output_file <ALIGNMENTS_OUTPUT_FILE_PATH>
```
If you have any issue regarding the DUC alignment regeneration, please contact via email.


MultiNews alignments are released in full.



## Data generation ##

Predicted alignments of MultiNews and CNN/DailyMail train and val datasets can be found [here](https://drive.google.com/drive/folders/1JnRrdbENzBLpbae5ZIKmil1fuZhm2toc?usp=sharing).

## Alignment model ##
To apply aligment model on your own data, follow the following steps:
  1. Download the trained model [here](https://drive.google.com/drive/folders/1kTaZQVxUm-RWbF71QpOue5xDuV7-IP2i?usp=sharing).

  2. Run
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
    
  4. To use the alignment model with your own data with different properties, you can inherent from the docSum2MRPC_Aligner class and overload the relevant functions.
