import os
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from dlomix.models import DetectabilityModel
from dlomix.constants import CLASSES_LABELS, alphabet, aa_to_int_dict
from dlomix.data import DetectabilityDataset
from config import load_args_reduce

def reduce_lib_random(lib,prop_to_keep,output_lib_path):
    lib,_ = load_lib(path=lib)

    lib = lib[~lib['Stripped.Sequence'].str.contains('X')]
    lib =lib[~lib['Stripped.Sequence'].str.contains('U')]
    seq = pd.unique(lib['Stripped.Sequence'])
    seq_random = np.random.choice(seq, size=int(seq.shape[0]*prop_to_keep), replace=False)
    df = pd.DataFrame(seq_random,columns=['Stripped.Sequence'])
    library_reduced = lib.join(other=df.set_index('Stripped.Sequence'), on='Stripped.Sequence', how='inner')
    library_reduced.to_parquet(output_lib_path, index=False)


def apply_model(args, list_seq):
    data = pd.DataFrame(list_seq, columns=['Sequences'])
    data['Label MaxLFQ'] = [0] * data.shape[0]
    data['Protein.Ids'] = [0] * data.shape[0]
    data.to_csv('temp.csv', index=False)

    print('Initialising model')
    ## Model init
    if args.model_type == 'Binary':
        total_num_classes = 2
    elif args.model_type == 'Multiclass':
        total_num_classes = len(CLASSES_LABELS)
    elif args.model_type == 'Regression':
        total_num_classes = 1
    else :
        raise ValueError('Invalid model type')

    num_cells = 64

    model = DetectabilityModel(num_units=num_cells, num_clases=total_num_classes)

    ## Loading model weights
    model.built = True

    model.load_weights(args.model_path)
    max_pep_length = 40
    ## Has no impact for prediction
    batch_size = 128

    print('Initialising dataset')
    ## Data init

    lib,_ = load_lib(args.base_lib_path)
    seq = pd.unique(lib['Stripped.Sequence'])
    data = pd.DataFrame(seq, columns=['Sequences'])
    data = data[~data['Sequences'].str.contains('X')]
    data['Label MaxLFQ'] = [0] * data.shape[0]
    data.to_csv('temp/temp.csv', index=False)

    detectability_data = DetectabilityDataset(data_source='temp/temp.csv',
                                              val_data_source='temp/temp.csv',
                                              data_format='csv',
                                              max_seq_len=max_pep_length,
                                              label_column="Label MaxLFQ",
                                              sequence_column="Sequences",
                                              dataset_columns_to_keep=None,
                                              batch_size=batch_size,
                                              with_termini=False,
                                              alphabet=aa_to_int_dict)
    val_data = detectability_data.tensor_val_data
    seq = detectability_data["val"]["_parsed_sequence"]
    seq = list(map(lambda x: "".join(x), seq))

    print('Applying model')
    ## Applying model
    predictions = model.predict(val_data)
    if args.model_type == 'Binary':
        result = pd.DataFrame(
            {'Sequences': seq, 'Prediction': predictions[:, 1]}) #Flyer probability
    elif args.model_type =='Regression':
        result = pd.DataFrame(
            {'Sequences': seq,
             'Prediction': predictions[:, 0]}) #Flyer intensity
    elif args.model_type =='Multiclass':
        result = pd.DataFrame(
            {'Sequences': seq,
             'Prediction': 1 - predictions[:, 0]})

    os.remove('temp/temp.csv')

    return result

def filter_lib(results,args,lib,schema):

    flyer_index = results[['Sequences', 'Prediction']]
    flyer_index = flyer_index.sort_values(by=['Prediction'], ascending=False)  # A vérifier
    last_row = flyer_index.shape[0] - 1
    ind = int((100 - args.percentage_to_drop) * last_row / 100)
    reduced_seq = flyer_index.iloc[:ind]
    library_reduced = lib.join(other=reduced_seq.set_index('Sequences'), on='Stripped.Sequence', how='inner')
    library_reduced = library_reduced.drop(columns='Prediction')
    library_reduced.to_parquet(args.output_lib_path, index=False,schema=schema)

def load_lib(path):
    table = pq.read_table(path)
    schema = table.schema
    table_pd = table.to_pandas()

    return table_pd, schema


def main():
    args = load_args_reduce()
    lib,schema = load_lib(path=args.base_lib_path)
    seq = pd.unique(lib['Stripped.Sequence'])
    results = apply_model(args=args, list_seq=seq)
    filter_lib(results=results,args=args,lib=lib,schema=schema)

if __name__=='__main__':
    # for prop in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    #     # reduce_lib_random('input_data/zeno/250723_proteines_ribosomales_chaperons_microbiote_lib.parquet',prop,'random_lib/random_lib_no_UX_{}.parquet'.format(prop))
    main()