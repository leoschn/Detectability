import argparse


def load_args_reduce():
    parser = argparse.ArgumentParser()
    # Input diann library path, must be previously converted to .parquet
    parser.add_argument('--base_lib_path', type=str, default='input_data/zeno/250723_proteines_ribosomales_chaperons_microbiote_lib.parquet')
    # Detectability model path (weight)
    parser.add_argument('--model_path', type=str, default='output/model/saved_model_binary_2.pt')
    # Type of detectability model, binary and regression are supported
    parser.add_argument('--model_type', type=str, default='Binary')
    # Percentage of the library to filter out based on predicted detectability (between 0 and 100)
    parser.add_argument('--percentage_to_drop', type=float, default=90.)
    # Output path of the reduced library
    parser.add_argument('--output_lib_path', type=str, default='output/reduced_lib_binary_finetuned_10_2.parquet')


    args = parser.parse_args()

    return args


def load_args_generate():
    parser = argparse.ArgumentParser()
    # Diann report file for detectability extraction (report.tsv file)
    parser.add_argument('--input_diann', type=str, default='input_data/zeno/ZENO_dataset_report.tsv')
    # Fasta file used for diann analysis
    parser.add_argument('--input_fasta', type=str, default='input_data/zeno/250325_17_proteomes_gut_std_ozyme_+_conta.fasta')
    # Type of label to be used in the dataset
    parser.add_argument('--label_type', type=str, default='Regression')
    # Type of quant used (frag prec or maxlfq)
    parser.add_argument('--type_quant', type=str, default='prec')
    # Fraction of non flyer in the train set default to 1 (4 balanced classes for multiclass, 2 balanced classes for binary)
    parser.add_argument('--frac_no_fly_train', type=float, default=1.)
    # Fraction of non flyer in the val set default to 1 (4 balanced classes for multiclass, 2 balanced classes for binary)
    parser.add_argument('--frac_no_fly_val', type=float, default=1.)
    # Fraction of non flyer in the test set default to 1 (4 balanced classes for multiclass, 2 balanced classes for binary)
    parser.add_argument('--frac_no_fly_test', type=float, default=1.)
    # Train/Test split size ratio, independently of class repartition
    parser.add_argument('--frac_split', type=tuple, default=(0.8,0.1,0.1))
    # Saving path for the train dataset
    parser.add_argument('--output_dataset_train_path', type=str, default='output_review/dataset_train_temp.csv')
    # Saving path for the val dataset
    parser.add_argument('--output_dataset_val_path', type=str, default='output_review/dataset_val_temp.csv')
    # Saving path for the test dataset
    parser.add_argument('--output_dataset_test_path', type=str, default='output_review/dataset_temp.csv')
    # Minimal proportion of the protein sequence covered by identified peptides to pass filter
    parser.add_argument('--coverage_threshold', type=float, default=0.2)
    # Minimal number of peptide identified by protein to pass filter
    parser.add_argument('--min_peptide', type=int, default=4)
    #minimal length of peptides to be included in the dataset (should be the same as the one used in dia-NN analysis)
    parser.add_argument('--min_len_pep', type=int, default=5)
    #reduce type for comuting quant from precuros inos intensity (max or sum)
    parser.add_argument('--reduce', type=str, default='max')
    # Seed for dataset shuffling
    parser.add_argument('--manual_seed', type=int, default=42)


    args = parser.parse_args()

    return args


def load_args_finetune():
    parser = argparse.ArgumentParser()
    # number of training epochs
    parser.add_argument('--epochs', type=int, default=50)
    # Type of training (must be compatible with dataset type)
    parser.add_argument('--task', type=str, default='Regression')
    parser.add_argument('--lr', type=float, default=0.005)
    # Path to the train dataset
    parser.add_argument('--path_dataset_train', type=str, default='output_review/zeno/dataset/ZENO_train_dataset_2.csv')
    # Path to validation dataset
    parser.add_argument('--path_dataset_val', type=str, default='output_review/zeno/dataset/ZENO_val_dataset_2.csv')
    # Path to test dataset
    parser.add_argument('--path_dataset_test', type=str, default='output_review/zeno/dataset/ZENO_test_dataset_2.csv')
    # Path to pretrained model weights
    parser.add_argument('--path_model', type=str, default='pretrained_model/original_detectability_fine_tuned_model_FINAL')
    # Path to save fine-tuned model weights
    parser.add_argument('--save_model_path', type=str, default='output_review/zeno/model/saved_model_test_lr_multi_5e-2.pt')
    parser.add_argument('--batch_size', type=int, default=16)
    # Path to the training and validation metrics report
    parser.add_argument('--report_path', type=str, default='output_review/zeno/report/report_test_lr_zeno_multiclass_5e-2')
    # Name of the model in training/validation report
    parser.add_argument('--model_name', type=str, default='model_test_lr_zeno_multiclass_5e-2')


    args = parser.parse_args()

    return args
