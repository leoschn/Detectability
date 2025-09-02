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


def build_dataset(coverage_treshold = 20, min_peptide = 4, f_name='out_df.csv',input_path=None, label_type='Binary'):
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

        df_final = pd.DataFrame.from_dict(dico_final, orient='index',
                                          columns=['Classes MaxLFQ'])
        df_final['Sequences'] = df_final.index
        df_final = df_final.reset_index()
        df_final = df_final[['Sequences', 'Classes MaxLFQ']]
        df_final.to_csv(f_name, index=False)
        df_non_flyer.to_csv('ISA_data/df_non_flyer_no_miscleavage.csv', index=False)

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

        df_final = pd.DataFrame.from_dict(dico_final, orient='index',
                                          columns=['Classes MaxLFQ'])
        df_final['Sequences'] = df_final.index
        df_final = df_final.reset_index()
        df_final = df_final[['Sequences', 'Classes MaxLFQ']]
        df_final.to_csv(f_name, index=False)
        df_non_flyer.to_csv('ISA_data/df_non_flyer_no_miscleavage.csv', index=False)


    elif label_type=='Regression':

        seq = df_grouped['Stripped.Sequence'].to_list()

        value_maxlfq = df_grouped['MaxLFQ'].to_list()


        max_max_lfq = max(value_maxlfq)
        for i in range(len(seq)):
            label_maxlfq = value_maxlfq[i] / max_max_lfq
            dico_final[seq[i]] =  label_maxlfq

        df_final = pd.DataFrame.from_dict(dico_final, orient='index',
                                          columns=['Value MaxLFQ'])
        df_final['Sequences'] = df_final.index
        df_final = df_final.reset_index()
        df_final = df_final[['Sequences', 'Value MaxLFQ']]
        df_final.to_csv('ISA_data/datasets/df_flyer_zeno_reg.csv', index=False)
        df_non_flyer.to_csv('ISA_data/datasets/df_non_flyer_zeno_reg.csv', index=False)

    else :

        raise Exception('Label type not supported')


if __name__ == '__main__':
    build_dataset()