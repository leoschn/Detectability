import os
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from dlomix.models import DetectabilityModel
from dlomix.constants import CLASSES_LABELS, alphabet, aa_to_int_dict
from dlomix.data import DetectabilityDataset
from config import load_args_reduce


def apply_model(model_path, model_type, list_seq):
    #Only compatible with binary and regression model
    data = pd.DataFrame(list_seq, columns=['Sequences'])
    data['Labels MaxLFQ'] = [0] * data.shape[0]
    data['Protein.Ids'] = [0] * data.shape[0]
    data.to_csv('temp.csv', index=False)

    print('Initialising model')
    ## Model init
    total_num_classes = len(CLASSES_LABELS)
    input_dimension = len(alphabet)
    num_cells = 64

    model = DetectabilityModel(num_units=num_cells, num_clases=total_num_classes)

    ## Loading model weights
    model.built = True

    model.load_weights(model_path)
    max_pep_length = 40
    ## Has no impact for prediction
    batch_size = 128

    print('Initialising dataset')
    ## Data init
    detectability_data = DetectabilityDataset(data_source='temp/temp.csv',
                                              val_data_source='temp/temp.csv',
                                              data_format='csv',
                                              max_seq_len=max_pep_length,
                                              label_column="Labels MaxLFQ",
                                              sequence_column="Sequences",
                                              dataset_columns_to_keep=['Protein.Ids'],
                                              batch_size=batch_size,
                                              with_termini=False,
                                              alphabet=aa_to_int_dict)
    val_data = detectability_data.tensor_val_data
    seq = detectability_data["val"]["_parsed_sequence"]
    seq = list(map(lambda x: "".join(x), seq))

    print('Applying model')
    ## Applying model
    if args.model_type == 'Binary':
        predictions = model.predict(val_data)
        label_binary = np.argmax(predictions, axis=1)
        result = pd.DataFrame(
            {'Sequences': seq, 'Prediction': predictions[:, 1]}) #Flyer probability
    elif args.model_type =='Regression':
        predictions = model.predict(val_data)
        result = pd.DataFrame(
            {'Sequences': seq,
             'Prediction': predictions}) #Flyer intensity
    else :
        raise Exception('Model type not supported')
    os.remove('temp.csv')

    return result

def load_lib(path):
    table = pq.read_table(path)
    table = table.to_pandas()

    return table

if __name__=='__main__':
    args = load_args_reduce()
    lib = load_lib(path=args.base_lib_path)
    seq = pd.unique(lib['Stripped.Sequence'])
    results = apply_model(model_path=args.model_path, model_type=args.model_type, list_seq=seq)
    flyer_index = results[['Sequences', 'Prediction']]
    flyer_index = flyer_index.sort_values(by=['Prediction'],ascending=False) #A vérifier
    last_row = flyer_index.shape[0]-1
    ind = int(100*last_row/(100-args.percentage_to_drop))
    reduced_seq = flyer_index.iloc[:ind]
    library_reduced = lib.join(other=reduced_seq.set_index('Sequences'),on='Stripped.Sequence',how='inner')
    library_reduced = library_reduced.drop(columns='Prediction')
    library_reduced.to_parquet(args.output_lib_path,index=False)
