import os
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from dlomix.models import DetectabilityModel
from dlomix.constants import CLASSES_LABELS, alphabet, aa_to_int_dict
from dlomix.data import DetectabilityDataset
from config import load_args_reduce


def apply_model(model_path, list_seq):
    data = pd.DataFrame(list_seq, columns=['Sequences'])
    data['Classes'] = [0] * data.shape[0]
    data['Proteins'] = [0] * data.shape[0]
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
    detectability_data = DetectabilityDataset(data_source='temp.csv',
                                              val_data_source='temp.csv',
                                              data_format='csv',
                                              max_seq_len=max_pep_length,
                                              label_column="Classes",
                                              sequence_column="Sequences",
                                              dataset_columns_to_keep=['Proteins'],
                                              batch_size=batch_size,
                                              with_termini=False,
                                              alphabet=aa_to_int_dict)
    val_data = detectability_data.tensor_val_data
    seq = detectability_data["val"]["_parsed_sequence"]
    seq = list(map(lambda x: "".join(x), seq))

    print('Applying model')
    ## Applying model
    predictions = model.predict(val_data)
    label_binary = np.argmax(predictions, axis=1)
    result = pd.DataFrame(
        {'Sequences': seq, 'Probability no flyer': predictions[:, 0], 'Probability flyer': predictions[:, 1],
         'Predicted class': label_binary})

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
    results = apply_model(model_path=args.model_path,list_seq=seq)
    prob_no_flyer = results[['Sequences', 'Probability no flyer']]
    prob_no_flyer = prob_no_flyer.sort_values(by=['Probability no flyer'],ascending=False)
    last_row = prob_no_flyer.shape[0]-1
    ind = int(100*last_row/args.percentage_to_drop)
    sliced_seq = prob_no_flyer.iloc[ind:-1]
    library_reduced = lib.join(other=sliced_seq.set_index('Sequences'),on='Stripped.Sequence',how='inner')
    library_reduced = library_reduced.drop(columns='Probability no flyer')
    library_reduced.to_parquet(args.output_lib_path,index=False)
