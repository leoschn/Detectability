import pandas as pd
from config import load_args_generate
"""
target_labels = {
    0: "Non-Flyer",
    1: "Weak Flyer",
    2: "Intermediate Flyer",
    3: "Strong Flyer",
}
binary_labels = {0: "Non-Flyer", 1: "Flyer"}
"""


def build_dataset(coverage_treshold, min_peptide, output_dataset_train_path,output_dataset_test_path,input_path, label_type,frac_split,frac_no_fly_train,frac_no_fly_test,manual_seed):
    df = pd.read_excel('ISA_data/250326_gut_microbiome_std_17_proteomes_data_training_detectability.xlsx')
    df_non_flyer = pd.read_csv('ISA_data/250422_FASTA_17_proteomes_gut_std_ozyme_+_conta_peptides_digested_filtered.csv')
    #No flyer
    df_non_flyer = df_non_flyer[df_non_flyer['Cystein ? ']=='Any']
    df_non_flyer = df_non_flyer[df_non_flyer['Miscleavage ?'] == 'Any']
    df_non_flyer = df_non_flyer[df_non_flyer['MaxLFQ'] == 0.0]
    df_non_flyer['Sequences'] = df_non_flyer['Peptide']
    df_non_flyer['Proteins'] = df_non_flyer['ProteinID']
    df_non_flyer=df_non_flyer[['Sequences','Proteins']].drop_duplicates()
    df_non_flyer['Classes fragment']=0
    df_non_flyer['Classes precursor'] =0
    df_non_flyer['Classes MaxLFQ'] =0


    #Flyer
    df_filtered = df[df['Proteotypic ?']=='Proteotypic']
    df_filtered = df_filtered[df_filtered['Coverage ']>=coverage_treshold]
    df_filtered = df_filtered[df_filtered['Miscleavage ?']=='Any']
    peptide_count=df_filtered.groupby(["Protein.Names"]).size().reset_index(name='counts')
    filtered_sequence = peptide_count[peptide_count['counts']>=min_peptide]["Protein.Names"]
    df_filtered = df_filtered[df_filtered["Protein.Names"].isin(filtered_sequence.to_list())]

    df_grouped = df_filtered.groupby("Protein.Names")
    dico_final={}

    if label_type=='Multi_class':

        # iterate over each group
        for group_name, df_group in df_grouped:
            seq = df_group.sort_values(by=['MaxLFQ'])['Stripped.Sequence'].to_list()

            for i in range(len(seq)):
                if i < int(len(seq) / 3):
                    label_maxlfq = 1
                elif i < int(2 * len(seq) / 3):
                    label_maxlfq = 2
                else:
                    label_maxlfq = 3

                dico_final[seq[i]] = label_maxlfq

        df_flyer = pd.DataFrame.from_dict(dico_final, orient='index',
                                          columns=['Classes MaxLFQ'])
        df_flyer['Sequences'] = df_flyer.index
        df_flyer = df_flyer.reset_index()
        df_flyer = df_flyer[['Sequences', 'Classes MaxLFQ']]

        # stratified split
        list_train_split = []
        list_val_split = []
        total_count = 0
        for cl in [1, 2, 3]:
            df_class = df_flyer[df_flyer['Classes MaxLFQ'] == cl]
            class_count = df_class.shape[0]
            list_train_split.append(df_class.iloc[:int(class_count * frac_split[0]), :])
            list_val_split.append(df_class.iloc[int(class_count * frac_split[0]):, :])
            total_count += class_count
        total_count = total_count / 3
        list_train_split.append(df_non_flyer.iloc[:int(total_count * frac_split[0] * frac_no_fly_train), :])
        list_val_split.append(
            df_non_flyer.iloc[df_non_flyer.shape[0] - int(total_count * frac_split[1] * frac_no_fly_test):, :])

        df_train = pd.concat(list_train_split).sample(frac=1, random_state=manual_seed)  # shuffle
        df_test = pd.concat(list_val_split).sample(frac=1, random_state=manual_seed)  # shuffle

        df_train['Proteins'] = 0
        df_test['Proteins'] = 0
        df_train.to_csv(output_dataset_train_path, index=False)
        df_test.to_csv(output_dataset_test_path, index=False)



    elif label_type=='Binary':

        # iterate over each group
        for group_name, df_group in df_grouped:
            seq = df_group.sort_values(by=['MaxLFQ'])['Stripped.Sequence'].to_list()

            for i in range(len(seq)):
                if i < int(len(seq) / 3):
                    label_maxlfq = 1
                elif i < int(2 * len(seq) / 3):
                    label_maxlfq = 2
                else:
                    label_maxlfq = 3

                dico_final[seq[i]] = label_maxlfq

        df_flyer = pd.DataFrame.from_dict(dico_final, orient='index',
                                          columns=['Classes MaxLFQ'])
        df_flyer['Sequences'] = df_flyer.index
        df_flyer = df_flyer.reset_index()
        df_flyer = df_flyer[['Sequences', 'Classes MaxLFQ']]


        #split
        list_train_split = []
        list_val_split = []
        flyer_count = df_flyer.shape[0]
        list_train_split.append(df_flyer.iloc[:int(flyer_count * frac_split[0]), :])
        list_val_split.append(df_flyer.iloc[int(flyer_count * frac_split[0]):, :])
        list_train_split.append(df_non_flyer.iloc[:int(flyer_count * frac_split[0] * frac_no_fly_train), :])
        list_val_split.append(
            df_non_flyer.iloc[df_non_flyer.shape[0] - int(flyer_count * frac_split[1] * frac_no_fly_test):, :])

        df_train = pd.concat(list_train_split).sample(frac=1, random_state=manual_seed)  # shuffle
        df_test = pd.concat(list_val_split).sample(frac=1, random_state=manual_seed)



        df_train.to_csv(output_dataset_train_path, index=False)
        df_test.to_csv(output_dataset_test_path, index=False)



    elif label_type=='Regression':

        seq = df_grouped['Stripped.Sequence'].to_list()

        value_maxlfq = df_grouped['MaxLFQ'].to_list()


        max_max_lfq = max(value_maxlfq)
        for i in range(len(seq)):
            label_maxlfq = value_maxlfq[i] / max_max_lfq
            dico_final[seq[i]] =  label_maxlfq

        df_flyer = pd.DataFrame.from_dict(dico_final, orient='index',
                                          columns=['Value MaxLFQ'])
        df_flyer['Sequences'] = df_flyer.index
        df_flyer = df_flyer.reset_index()
        df_flyer = df_flyer[['Sequences', 'Value MaxLFQ']]


        #split
        #split
        list_train_split = []
        list_val_split = []
        flyer_count = df_flyer.shape[0]
        list_train_split.append(df_flyer.iloc[:int(flyer_count * frac_split[0]), :])
        list_val_split.append(df_flyer.iloc[int(flyer_count * frac_split[0]):, :])
        list_train_split.append(df_non_flyer.iloc[:int(flyer_count * frac_split[0] * frac_no_fly_train), :])
        list_val_split.append(
            df_non_flyer.iloc[df_non_flyer.shape[0] - int(flyer_count * frac_split[1] * frac_no_fly_test):, :])

        df_train = pd.concat(list_train_split).sample(frac=1, random_state=manual_seed)  # shuffle
        df_test = pd.concat(list_val_split).sample(frac=1, random_state=manual_seed)
        df_train.to_csv(output_dataset_train_path, index=False)
        df_test.to_csv(output_dataset_test_path, index=False)



    else :

        raise Exception('Label type not supported')


if __name__ == '__main__':
    args = load_args_generate()
    build_dataset(coverage_threshold=args.coverage_threshold, min_peptide=args.min_peptide,
                  input_path=args.diann_report_matrix_path,label_type=args.label_type,
                  output_dataset_train_path=args.output_dataset_train_path
                  ,output_dataset_test_path=args.output_dataset_test_path)