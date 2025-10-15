# Detectability extraction and fine tuning

Implementation of Generation of peptide detectability datasets from
single DIA spectra for prediction model fine-tuning paper. It includes :
- Extraction of detectability dataset from [DiaNN](https://github.com/vdemichev/DiaNN) analysis of a single DIA run
- Finetuning of [pFly](https://github.com/wilhelm-lab/dlomix] model) with binary categorical or regression task
- reduction of a DiaNN search library based on predicted detectability

## Installation

Compatible python package can be installed using :

    pip install -r requirements.txt

## Usage

The following command will extract detectability from a DIANN report and create a finetuning dataset based on it.

    python generate_dataset.py --input_diann path/to/report.tsv --input_fasta path/to/fasta --label_type [Binary,Multi_class,Regression] --output_dataset_train_path path/to/output_train --output_dataset_test_path path/to/output_test

To fine tune pFLy model using a previously generated dataset you can use :

    python finetune_model.py --task [Binary,Multi_class,Regression] --path_dataset_train path/to/output_train --path_dataset_test path/to/output_test  --path_model path/to/base_model --save_model_path path/to/seve_finetuned_model --report_path path/to/finetuning_report

Finally, to reduce a DIANN library using a detectability model use :

    python reduce_library.py --base_lib_path path/to/base_lib --model_path path/to/detectability_model --model_type [Binary,Multi_class,Regression]  --percentage_to_drop [0-100] --output_lib_path path/to/output_lib

Additional and default arguments for each function are described in config.py
