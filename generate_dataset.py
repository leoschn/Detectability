import pandas as pd
import numpy as np
from config import load_args_generate
from Bio import SeqIO
from collections import defaultdict


"""
target_labels = {
    0: "Non-Flyer",
    1: "Weak Flyer",
    2: "Intermediate Flyer",
    3: "Strong Flyer",
}
binary_labels = {0: "Non-Flyer", 1: "Flyer"}
"""

def tryptic_digest(sequence, miscleavages=0, min_length=0, max_length=10000):
    cleavage_sites = [0]
    for i in range(len(sequence) - 1):
        if sequence[i] in ['K', 'R'] and sequence[i+1] != 'P':
            #Règle : clivage après K ou R sauf si suivi de P.
            cleavage_sites.append(i + 1)
    cleavage_sites.append(len(sequence))

    peptides = []
    for i in range(len(cleavage_sites) - 1):
        for j in range(i + 1, min(i + miscleavages + 2, len(cleavage_sites))):
            start = cleavage_sites[i]
            end = cleavage_sites[j]
            peptide = sequence[start:end]
            if min_length <= len(peptide) <= max_length:
                peptides.append(peptide)
    return peptides

def digest_fasta(input_fasta, miscleavages=0, min_length=0, max_length=10000):
    all_peptides = []
    for record in SeqIO.parse(input_fasta, "fasta"):
        seq = str(record.seq)
        peptides = tryptic_digest(seq, miscleavages=miscleavages, min_length=min_length, max_length=max_length)
        for peptide in peptides:
            all_peptides.append({
                'ProteinID': record.id,
                'Peptide': peptide
            })

    return all_peptides

def load_fasta_sequences(fasta_file):
    """Charge les séquences du fichier FASTA et retourne un dictionnaire {ID_protéine: séquence}"""
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences[record.id] = str(record.seq)
    return sequences

# Fonction pour compter les sites de clivage (lysine 'K' ou arginine 'R')
def count_cleavages(peptide_sequence):
    # Compter le nombre de résidus K ou R dans la séquence du peptide
    return peptide_sequence[:-1].count('K') + peptide_sequence[:-1].count('R') -peptide_sequence.count('KP') - peptide_sequence.count('RP')


def find_proteotypic_peptides(peptide_list, fasta_file):
    """Optimisé pour vérifier si les peptides sont protéotypiques dans un fichier FASTA."""
    # Charger les séquences des protéines
    protein_sequences = load_fasta_sequences(fasta_file)


    # Dictionnaire pour associer chaque peptide aux protéines où il apparaît
    peptide_to_protein = defaultdict(set)

    # Une seule boucle pour vérifier tous les peptides dans chaque protéine
    for protein_id, sequence in protein_sequences.items():
        for peptide in peptide_list:
            if peptide in sequence:
                peptide_to_protein[peptide].add(protein_id)

    # Identification des peptides protéotypiques (qui apparaissent dans une seule protéine)
    proteotypic_peptides = {pep for pep, proteins in peptide_to_protein.items() if len(proteins) == 1}

    return proteotypic_peptides

def calculate_sequence_coverage(fasta_file, peptide_list):
    """Calcule la couverture de séquence des protéines à partir des peptides et exporte les résultats dans un fichier CSV."""
    protein_sequences = load_fasta_sequences(fasta_file)

    # Dictionnaire pour stocker la couverture des protéines
    protein_coverage = {}

    for protein_id, sequence in protein_sequences.items():
        # Liste des positions couvertes par les peptides
        covered_positions = set()

        # Vérification des peptides dans chaque protéine
        for peptide in peptide_list:
            start = 0
            while start < len(sequence):
                start = sequence.find(peptide, start)
                # Ajouter les positions couvertes par le peptide
                covered_positions.update(range(start, start + len(peptide)))
                start += 1

        # Calcul de la couverture
        coverage = (len(covered_positions) / len(sequence)) * 100
        protein_coverage[protein_id] = coverage

    return protein_coverage

def build_dataset(coverage_threshold, min_peptide, output_dataset_train_path,output_dataset_test_path,input_fasta,
                  input_id, label_type,frac_split,frac_no_fly_train,frac_no_fly_test,manual_seed):
    peptides_identification = pd.read_csv(input_id, sep="\t")
    #build dataset
    id_peptide_list = peptides_identification['Stripped.Sequence'].unique().tolist()
    all_peptides = [x['peptide'] for x in digest_fasta(input_fasta,miscleavages=0)]
    non_flyer_list = list(set(all_peptides) - set(id_peptide_list))
    proteotypic_peptides = find_proteotypic_peptides(input_fasta, id_peptide_list)
    protein_coverage = calculate_sequence_coverage(input_fasta, id_peptide_list)
    df_flyer = peptides_identification[['Stripped.Sequence', 'Protein.Names','PG.MaxLFQ']].drop_duplicates()
    df_flyer['Contains Cystein']=np.where('C' in df_flyer['Stripped.Sequence'])
    df_flyer['Proteotypic']=np.where(df_flyer['Stripped.Sequence'] in proteotypic_peptides)
    df_flyer['Coverage']=np.where(df_flyer['Proteotypic'],protein_coverage[df_flyer['Stripped.Sequence']],0)
    df_flyer['Miscleavage'] = df_flyer['Stripped.Sequence'].apply(count_cleavages)
    peptide_count = df_flyer.groupby(["Protein.Names"]).size().reset_index(name='counts')

    #filter dataset
    filtered_sequence = peptide_count[peptide_count['counts'] >= min_peptide]["Protein.Names"]
    df_flyer = df_flyer[df_flyer["Protein.Names"].isin(filtered_sequence.to_list())]
    df_flyer=df_flyer[df_flyer['Proteotypic']==True]
    df_flyer = df_flyer[df_flyer['Coverage'] >= coverage_threshold]
    df_flyer = df_flyer[df_flyer['Miscleavage'] == 0]
    df_flyer = df_flyer[df_flyer['Contains Cystein'] == False]



    #No flyer

    df_non_flyer = pd.DataFrame(non_flyer_list, columns=['Stripped.Sequence'])
    df_non_flyer['Contains Cystein']=np.where('C' in df_non_flyer['Stripped.Sequence'])
    df_non_flyer['Miscleavage'] = df_non_flyer['Stripped.Sequence'].apply(count_cleavages)
    df_non_flyer = df_non_flyer[df_non_flyer['Contains Cystein']==False]
    df_non_flyer = df_non_flyer[df_non_flyer['Miscleavage'] == 0]
    df_non_flyer['Sequences'] = df_non_flyer['Stripped.Sequence']
    df_non_flyer=df_non_flyer[['Sequences']].drop_duplicates()
    df_non_flyer['Classes MaxLFQ'] =0

    #compute labels and split datasets
    df_grouped = df_flyer.groupby("Protein.Names")
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
        df_flyer=df_flyer['Sequences']
        df_flyer['Classes MaxLFQ']=1
        df_flyer = df_flyer.reset_index()
        df_flyer = df_flyer[['Sequences', 'Classes MaxLFQ']]
        df_non_flyer=df_non_flyer[['Sequences', 'Classes MaxLFQ']]


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
                  input_id=args.input_diann, input_fasta=args.input_fasta,label_type=args.label_type,
                  output_dataset_train_path=args.output_dataset_train_path,
                  output_dataset_test_path=args.output_dataset_test_path)