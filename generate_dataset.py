import pandas as pd
import ahocorasick
import numpy as np
from config import load_args_generate
from Bio import SeqIO
from collections import defaultdict

# Using R inside python
from rpy2.robjects.vectors import StrVector
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

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
    """
    In Silico digest proteins from fasta file into peptides following trypsin cleavage rules

    Args:
    input_fasta (str) : path of the fasta file
    miscleavages (int) : maximal number of miscleavages allowed
    min_length (int): minimal length of peptides
    max_length (int): maximal length of peptides

    Returns:
        dict: {protein,peptides list}
    """
    all_peptides = []
    for record in SeqIO.parse(input_fasta, "fasta"):
        seq = str(record.seq)
        peptides = tryptic_digest(seq, miscleavages=miscleavages, min_length=min_length, max_length=max_length)
        for peptide in peptides:
            all_peptides.append({
                'Protein.Ids': record.id,
                'Stripped.Sequence': peptide
            })

    return all_peptides

def load_fasta_sequences(fasta_file):
    """
     Load sequences from a fasta file.

     Agrs:
     fasta_file (str): fasta file path

     Returns:
        dict : {protein id: sequence}
     """
    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        if '#' in record.id:
            sequences[record.id.split('#')[-1]] = str(record.seq)
        elif "|" in record.id:
            sequences[record.id.split('|')[1]] = str(record.seq)
    return sequences

# Fonction pour compter les sites de clivage (lysine 'K' ou arginine 'R')
def count_cleavages(peptide_sequence):
    """
    Count cleavage site in the sequence
    Args:
     peptide_sequence (str): Amino acid sequence

    Returns:
        int : cleavage site count
    """
    # Compter le nombre de résidus K ou R dans la séquence du peptide
    return peptide_sequence[:-1].count('K') + peptide_sequence[:-1].count('R') -peptide_sequence.count('KP') - peptide_sequence.count('RP')


def find_proteotypic_peptides(peptide_list, fasta_file):
    """
    Filter proteotypic peptides from a list of peptides and a fasta file

    Args:
        peptide_list (list): list of peptide sequences
        fasta_file (path): fasta file path

    Returns:
        dict: {peptide sequence: protein}
    """
    # Charger les séquences des protéines
    print('Finding proteotypic peptides...')
    protein_sequences = load_fasta_sequences(fasta_file)
    print('Protein sequences loaded.')

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

def build_automaton(peptides: list[str]):
    """
    Build an Aho-Corasick automaton for fast multi-peptide searching.
    """
    automaton = ahocorasick.Automaton()
    for idx, pep in enumerate(peptides):
        automaton.add_word(pep, (idx, pep))
    automaton.make_automaton()
    return automaton

def protein_coverage(protein: str, automaton) -> float:
    """
    Compute proportion of residues in a protein covered by peptides
    using Aho-Corasick automaton.
    """
    covered = [False] * len(protein)
    for end_idx, (_, pep) in automaton.iter(protein):
        start_idx = end_idx - len(pep) + 1
        for i in range(start_idx, end_idx + 1):
            covered[i] = True
    return sum(covered) / len(protein) if len(protein) > 0 else 0.0


def multiple_protein_coverage(fasta_file: str, peptides: list[str]):
    """
    Compute coverage for multiple proteins with a shared peptide list.

    Args:
        fasta_file (path): fasta file path
        peptides (list): list of peptide sequences (shared across proteins)

    Returns:
        dict: {protein_id: coverage_fraction}
    """
    protein_sequences = load_fasta_sequences(fasta_file)
    automaton = build_automaton(peptides)
    results = {}
    for pid, seq in protein_sequences.items():
        results[pid] = protein_coverage(seq, automaton)
    return results

def build_dataset(coverage_threshold, min_peptide, output_dataset_train_path,output_dataset_val_path,output_dataset_test_path,input_fasta,
                  input_id, label_type,frac_split,frac_no_fly,type_quant,min_len_pep,reduce,manual_seed):

    """
    Builds and splits a peptide dataset into training and test sets for proteomics analysis.

    This function processes peptide identification data from a FASTA file and an input
    identification table, applies multiple filtering steps (coverage, proteotypic status,
    cysteine content, miscleavage, and minimum peptides per protein), and generates labeled
    datasets for downstream machine learning tasks. The labeling scheme depends on the
    specified `label_type` (multi-class, binary, or regression).

    The resulting datasets are split into training and testing sets according to the specified
    fractions and saved as CSV files.

    Args:
        coverage_threshold (float): Minimum protein coverage required to retain peptides.
        min_peptide (int): Minimum number of peptides per protein required for inclusion.
        output_dataset_train_path (str): File path to save the training dataset (CSV).
        output_dataset_val_path (str): File path to save the validation dataset (CSV).
        output_dataset_test_path (str): File path to save the test dataset (CSV).
        input_fasta (str): Path to the input FASTA file containing protein sequences.
        input_id (str): Path to the input diann report.tsv table.
        label_type (str): Type of labeling strategy. Options are:
            - "Multi_class": Assign peptides to three expression bins per protein.
            - "Binary": Flyer peptides are labeled `1`, non-flyer peptides `0`.
            - "Regression": Continuous labels normalized by maximum protein intensity.
        frac_split (Tuple[float, float, float]): Fractions for train/val/test split (e.g., (0.8, 0.1, 0.1)
        frac_no_fly (Tuple[float, float, float]): Proportion of non-flyer peptides to include in each set.
        manual_seed (int): Random seed for reproducible shuffling.

    Raises:
        Exception: If `label_type` is not one of ["Multi_class", "Binary", "Regression"].
        Exception: If `quant_type` is not one of ["frag", "prec", "maxlfq"].
        ValueError: Too few non-flyer to have requested non-flyer frac

    Outputs:
        - Saves the training dataset as a CSV at `output_dataset_train_path`.
        - Saves the test dataset as a CSV at `output_dataset_test_path`.

    Notes:
        - Flyer peptides are those detected in the experiment (from `input_id`).
        - Non-flyer peptides are generated by digesting the input FASTA and subtracting detected peptides.
        - Several peptides filtering rules are enforced:
            * Must be proteotypic.
            * Must not contain cysteine residues.
            * Must have zero miscleavages.
            * Must meet protein coverage and peptide count thresholds.
    """

    #build dataset
    print('Loading file')

    frac_no_fly_train,frac_no_fly_val,frac_no_fly_test = frac_no_fly
    if type_quant == 'maxlfq' :
        iq = importr('iq')
        annotation_col = StrVector(["Protein.Ids", "Genes"])
        filter_double_less = ro.ListVector({
            "Q.Value": "0.01",
            "PG.Q.Value": "0.05",
            "Lib.Q.Value": "0.01",
            "Lib.PG.Q.Value": "0.01"
        })

        # Compute maxLFQ of precursor and store it in temp dir
        #Pham TV, Henneman AA, Jimenez CR. iq: an R package to estimate relative protein abundances
        # from ion quantification in DIA-MS-based proteomics. Bioinformatics 2020 Apr 15;36(8):2611-
        # 2613.
        iq.process_long_format(input_id,
            sample_id="Run",
            primary_id="Stripped.Sequence",
            intensity_col="Fragment.Quant.Raw",
            output_filename="temp/report-maxlfq.txt",
            annotation_col=annotation_col,
            filter_double_less=filter_double_less
        )

        peptides_identification = pd.read_csv("temp/report-maxlfq.txt", sep="\t")
        quant_col = peptides_identification.columns[-1]

    elif type_quant == 'frag':
        report = pd.read_csv(input_id, sep="\t")
        report_grouped = report.groupby(['Stripped.Sequence', 'Protein.Ids'])
        list_quant = []
        for name, group in report_grouped:
            group['frag.quant']=group['Fragment.Quant.Raw'].map(lambda x: sum([float(y) for y in x.split(';')[:-1]]))
            if reduce == 'max':
                quant = group['frag.quant'].max()
            elif reduce == 'sum':
                quant = group['frag.quant'].sum()
            else :
                raise Exception('Unrecognized reduce type')
            list_quant.append({'Stripped.Sequence': name[0], 'Fragment.Quant': quant, 'Protein.Ids': name[1]})
        quant_col = 'Fragment.Quant'
        peptides_identification = pd.DataFrame(list_quant)

    elif type_quant =='prec':
        report = pd.read_csv(input_id, sep="\t")
        report_grouped = report.groupby(['Stripped.Sequence', 'Protein.Ids'])
        list_quant = []
        for name, group in report_grouped:
            if reduce == 'max':
                quant = group['Precursor.Quantity'].max()
            elif reduce == 'sum':
                quant = group['Precursor.Quantity'].sum()
            else :
                raise Exception('Unrecognized reduce type')
            list_quant.append({'Stripped.Sequence':name[0],'Precursor.Quantity':quant,'Protein.Ids':name[1]})
        quant_col = 'Precursor.Quantity'
        peptides_identification = pd.DataFrame(list_quant)
    else :
        raise Exception('Quantification type not supported')



    id_peptide_list = peptides_identification['Stripped.Sequence'].unique().tolist()
    digested_fasta = digest_fasta(input_fasta,miscleavages=0)
    non_flyer_list = [{'Stripped.Sequence':x['Stripped.Sequence'],'Protein.Ids':x['Protein.Ids']} for x in digested_fasta if x['Stripped.Sequence'] not in id_peptide_list]
    print('Searching for proteotypic peptides')
    proteotypic_peptides = find_proteotypic_peptides(id_peptide_list,input_fasta)
    print('Computing coverage')
    protein_coverage = multiple_protein_coverage(input_fasta, id_peptide_list)
    df_flyer = peptides_identification[['Stripped.Sequence', 'Protein.Ids',quant_col]].drop_duplicates()
    df_flyer = df_flyer[~df_flyer['Stripped.Sequence'].str.contains('X')]  # remove id with unknown aminio acid
    print('Filtering peptides')
    df_flyer['Contains Cystein']=df_flyer['Stripped.Sequence'].str.contains('C')
    df_flyer['Proteotypic']= df_flyer['Stripped.Sequence'].isin(proteotypic_peptides)
    df_flyer['Coverage'] = np.where(df_flyer['Proteotypic'],df_flyer['Protein.Ids'].map(protein_coverage),0)
    df_flyer['Miscleavage'] = df_flyer['Stripped.Sequence'].apply(count_cleavages)
    peptide_count = df_flyer.groupby(["Protein.Ids"]).size().reset_index(name='counts')

    #filter dataset
    filtered_sequence = peptide_count[peptide_count['counts'] >= min_peptide]["Protein.Ids"]
    df_flyer = df_flyer[df_flyer["Protein.Ids"].isin(filtered_sequence.to_list())]
    df_flyer=df_flyer[df_flyer['Proteotypic']==True]
    df_flyer = df_flyer[df_flyer['Coverage'] >= coverage_threshold]
    df_flyer = df_flyer[df_flyer['Miscleavage'] == 0]
    df_flyer = df_flyer[df_flyer['Contains Cystein'] == False]
    df_flyer = df_flyer[df_flyer['Stripped.Sequence'].map(lambda x:len(x))>=min_len_pep]

    df_flyer['Sequences'] = df_flyer['Stripped.Sequence']
    df_flyer['Label']=df_flyer[quant_col]



    #No flyer

    df_non_flyer = pd.DataFrame(non_flyer_list)
    df_non_flyer['Protein.Ids']=df_non_flyer['Protein.Ids'].map(lambda x: x.split('|')[1] if 'tr|' in x else x)
    df_non_flyer = df_non_flyer[~df_non_flyer['Stripped.Sequence'].str.contains('X')]
    df_non_flyer['Contains Cystein']=df_non_flyer['Stripped.Sequence'].str.contains('C')
    df_non_flyer['Miscleavage'] = df_non_flyer['Stripped.Sequence'].apply(count_cleavages)
    df_non_flyer = df_non_flyer[df_non_flyer['Contains Cystein']==False]
    df_non_flyer = df_non_flyer[df_non_flyer['Miscleavage'] == 0]
    df_non_flyer = df_non_flyer[df_non_flyer['Stripped.Sequence'].map(lambda x: len(x)) >= min_len_pep]
    df_non_flyer['Sequences'] = df_non_flyer['Stripped.Sequence']
    df_non_flyer=df_non_flyer[['Sequences','Protein.Ids']].drop_duplicates()
    df_non_flyer['Label'] =0
    df_non_flyer = df_non_flyer[['Sequences', 'Label', "Protein.Ids"]]

    #compute labels and split datasets
    df_grouped = df_flyer.groupby("Protein.Ids")


    if label_type=='Multi_class':
        dico_final = {}
        # iterate over each group
        for group_name, df_group in df_grouped:
            seq = df_group.sort_values(by=['Label'])['Stripped.Sequence'].to_list()
            for i in range(len(seq)):
                if i < int(len(seq) / 3):
                    label_maxlfq = 1
                elif i < int(2 * len(seq) / 3):
                    label_maxlfq = 2
                else:
                    label_maxlfq = 3

                dico_final[seq[i]] = [label_maxlfq,group_name]

        df_flyer = pd.DataFrame.from_dict(dico_final, orient='index',
                                          columns=['Label',"Protein.Ids"])
        df_flyer['Sequences'] = df_flyer.index

    elif label_type=='Multi_class_global':
        dico_final = {}
        # iterate over each group
        df_sorted = df_flyer.sort_values(by=['Label'])[['Stripped.Sequence','Protein.Ids']]
        seq = df_sorted['Stripped.Sequence'].to_list()
        prot = df_sorted['Protein.Ids'].to_list()
        for i in range(len(seq)):
            if i < int(len(seq) / 3):
                label_maxlfq = 1
            elif i < int(2 * len(seq) / 3):
                label_maxlfq = 2
            else:
                label_maxlfq = 3
            dico_final[seq[i]] = [label_maxlfq,prot[i]]

        df_flyer = pd.DataFrame.from_dict(dico_final, orient='index',
                                          columns=['Label',"Protein.Ids"])
        df_flyer['Sequences'] = df_flyer.index


    elif label_type=='Binary':
        df_flyer=df_flyer[['Sequences',"Protein.Ids"]].copy()
        df_flyer['Label']=1

    elif label_type=='Regression':
        dico_final = {}
        for group_name, df_group in df_grouped:
            seq = df_group['Stripped.Sequence'].to_list()
            value_maxlfq = df_group['Label'].to_list()
            max_max_lfq = max(value_maxlfq)
            for i in range(len(seq)):
                label_maxlfq = value_maxlfq[i] / max_max_lfq
                dico_final[seq[i]] =  [label_maxlfq,group_name]

        df_flyer = pd.DataFrame.from_dict(dico_final, orient='index',
                                          columns=['Label',"Protein.Ids"])
        df_flyer['Sequences'] = df_flyer.index

    else :

        raise Exception('Label type not supported')


    df_flyer = df_flyer.reset_index()
    df_flyer = df_flyer[['Sequences', 'Label', "Protein.Ids"]]

    protein_counts = (
        df_flyer
        .groupby("Protein.Ids")
        .size()
        .reset_index(name="peptide_count")
    )

    protein_counts = protein_counts.sample(frac=1, random_state=manual_seed).reset_index(drop=True)

    total_peptides = protein_counts["peptide_count"].sum()

    train_target = int(frac_split[0] * total_peptides)
    val_target = (frac_split[0] + frac_split[1]) * total_peptides
    test_target = (frac_split[0] + frac_split[1] + frac_split[2]) * total_peptides

    protein_counts["cumulative"] = protein_counts["peptide_count"].cumsum()

    train_proteins = protein_counts.loc[
        protein_counts["cumulative"] <= train_target,
        "Protein.Ids"
    ]

    val_proteins = protein_counts.loc[
        (protein_counts["cumulative"] > train_target) &
        (protein_counts["cumulative"] <= val_target),
        "Protein.Ids"
    ]

    test_proteins = protein_counts.loc[
        protein_counts["cumulative"] > val_target,
        "Protein.Ids"
    ]

    df_train_flyer = df_flyer[df_flyer["Protein.Ids"].isin(train_proteins)]
    df_val_flyer = df_flyer[df_flyer["Protein.Ids"].isin(val_proteins)]
    df_test_flyer = df_flyer[df_flyer["Protein.Ids"].isin(test_proteins)]

    # check for protein intersection
    print(set(df_train_flyer["Protein.Ids"]) & set(df_val_flyer["Protein.Ids"]))
    print(set(df_train_flyer["Protein.Ids"]) & set(df_test_flyer["Protein.Ids"]))
    print(set(df_val_flyer["Protein.Ids"]) & set(df_test_flyer["Protein.Ids"]))

    # check split ration
    print(len(df_train_flyer) / len(df_flyer))
    print(len(df_val_flyer) / len(df_flyer))
    print(len(df_test_flyer) / len(df_flyer))

    # add non-flyers
    if label_type == 'Multi_class' or 'Multi_class_global':
        non_flyer_train_target = int((frac_split[0] * total_peptides * frac_no_fly_train)/3)
        non_flyer_val_target = int((frac_split[1] * total_peptides * frac_no_fly_val)/3) + non_flyer_train_target
        non_flyer_test_target = int((frac_split[2] * total_peptides * frac_no_fly_test)/3) + non_flyer_val_target
    elif label_type == 'Regression' or label_type == 'Binary':
        non_flyer_train_target = int(frac_split[0] * total_peptides * frac_no_fly_train)
        non_flyer_val_target = int(frac_split[1] * total_peptides * frac_no_fly_val) + non_flyer_train_target
        non_flyer_test_target = int(frac_split[2] * total_peptides * frac_no_fly_test) + non_flyer_val_target
    else:
        raise Exception('Label type not supported')

    if len(df_non_flyer) < test_target * frac_no_fly_train:
        raise ValueError('To few non flyers')
    df_train_non_flyer = df_non_flyer.iloc[:non_flyer_train_target, :]
    df_val_non_flyer = df_non_flyer.iloc[non_flyer_train_target:non_flyer_val_target, :]
    df_test_non_flyer = df_non_flyer.iloc[non_flyer_val_target:non_flyer_test_target, :]
    # check if enougth non-flyer

    df_train = pd.concat([df_train_non_flyer, df_train_flyer]).sample(frac=1, random_state=manual_seed)  # shuffle
    df_val = pd.concat([df_val_non_flyer, df_val_flyer]).sample(frac=1, random_state=manual_seed)
    df_test = pd.concat([df_test_non_flyer, df_test_flyer]).sample(frac=1, random_state=manual_seed)

    df_train.to_csv(output_dataset_train_path, index=False)
    df_val.to_csv(output_dataset_val_path, index=False)
    df_test.to_csv(output_dataset_test_path, index=False)


if __name__ == '__main__':
    args = load_args_generate()
    build_dataset(coverage_threshold=args.coverage_threshold, min_peptide=args.min_peptide,
          input_id=args.input_diann,
          input_fasta=args.input_fasta,
          label_type=args.label_type,
          output_dataset_train_path=args.output_dataset_train_path,
          output_dataset_val_path=args.output_dataset_val_path,
          output_dataset_test_path=args.output_dataset_test_path,
          frac_no_fly=[args.frac_no_fly_train,args.frac_no_fly_val,args.frac_no_fly_test],
          frac_split=args.frac_split,
          manual_seed=args.manual_seed,
          type_quant=args.type_quant,
          min_len_pep=args.min_len_pep)

