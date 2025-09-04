import argparse


def load_args_reduce():
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_lib_path', type=str, default='input_data/lib_parquet')
    parser.add_argument('--model_path', type=str, default='output/model/binary_zeno')
    parser.add_argument('--percentage_to_drop', type=float, default=40.)
    parser.add_argument('--output_lib_path', type=str, default='output/reduced_lib.parquet')


    args = parser.parse_args()

    return args


def load_args_generate():
    parser = argparse.ArgumentParser()

    parser.add_argument('--diann_report_matrix_path', type=str, default='input_data/report_matrix.tsv')
    parser.add_argument('--label_type', type=str, default='Binary')
    parser.add_argument('--output_dataset_train_path', type=str, default='output/dataset_train.csv')
    parser.add_argument('--output_dataset_test_path', type=str, default='output/dataset_test.csv')
    parser.add_argument('--train_test_split', type=tuple, default=(0.8,0.2))
    parser.add_argument('--coverage_threshold', type=float, default=20)
    parser.add_argument('--min_peptide', type=int, default=4)


    args = parser.parse_args()

    return args


def load_args_finetune():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=str, default='input_data/report_matrix.tsv')
    parser.add_argument('--task', type=str, default='Binary')
    parser.add_argument('--path_dataset_train', type=str, default='output/dataset.csv')
    parser.add_argument('--path_dataset_val', type=str, default='output/dataset.csv')
    parser.add_argument('--path_model', type=str, default='output/dataset.csv')
    parser.add_argument('--path_saved_model', type=str, default='output/dataset.csv')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--report_path', type=str, default='output/report_path.pdf')
    parser.add_argument('--model_name', type=str, default='output/model_finetuned.pt')


    args = parser.parse_args()

    return args
