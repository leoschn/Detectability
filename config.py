import argparse


def load_args_reduce():
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_lib_path', type=str, default='input_data/lib_parquet')
    parser.add_argument('--model_path', type=str, default='output/model/binary_zeno')
    parser.add_argument('--model_type', type=str, default='Binary')
    parser.add_argument('--percentage_to_drop', type=float, default=40.)
    parser.add_argument('--output_lib_path', type=str, default='output/reduced_lib.parquet')


    args = parser.parse_args()

    return args


def load_args_generate():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_diann', type=str, default='input_data/zeno/report.tsv')
    parser.add_argument('--input_fasta', type=str, default='input_data/zeno/250325_17_proteomes_gut_std_ozyme_+_conta.fasta')
    parser.add_argument('--label_type', type=str, default='Multi_class')
    parser.add_argument('--frac_no_fly_train', type=float, default=1.)
    parser.add_argument('--frac_no_fly_test', type=float, default=1.)
    parser.add_argument('--frac_split', type=tuple, default=(0.8,0.2))
    parser.add_argument('--output_dataset_train_path', type=str, default='output/dataset_train_multi.csv')
    parser.add_argument('--output_dataset_test_path', type=str, default='output/dataset_test_multi.csv')
    parser.add_argument('--coverage_threshold', type=float, default=0.2)
    parser.add_argument('--min_peptide', type=int, default=4)
    parser.add_argument('--manual_seed', type=int, default=42)


    args = parser.parse_args()

    return args


def load_args_finetune():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--task', type=str, default='Multi_class')
    parser.add_argument('--path_dataset_train', type=str, default='output/dataset_train_multi.csv')
    parser.add_argument('--path_dataset_val', type=str, default='output/dataset_test_multi.csv')
    parser.add_argument('--path_model', type=str, default='pretrained_model/original_detectability_fine_tuned_model_FINAL')
    parser.add_argument('--path_saved_model', type=str, default='output/saved_model.pt')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--report_path', type=str, default='output/report_path_multi')
    parser.add_argument('--model_name', type=str, default='output/model_finetuned_multi.pt')


    args = parser.parse_args()

    return args
