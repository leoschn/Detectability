import numpy as np
import pandas as pd
import tensorflow as tf
from dlomix.models import DetectabilityModel
from dlomix.constants import CLASSES_LABELS, aa_to_int_dict
from dlomix.data import DetectabilityDataset
from dlomix.reports.DetectabilityReport import DetectabilityReport
from config import load_args_finetune

if __name__ == "__main__":

    args = load_args_finetune()

    num_cells = 64
    total_num_classes = len(CLASSES_LABELS)


    if args.task =='Multi':
        total_num_classes = len(CLASSES_LABELS)
        fine_tuned_model = DetectabilityModel(num_units=num_cells,
                                              num_clases=total_num_classes)

        fine_tuned_model.load_weights(args.path_model)
    elif args.task =='Binary':

        fine_tuned_model = DetectabilityModel(num_units=num_cells,
                                              num_clases=2)
        fine_tuned_model.build((None, 40))

        base_arch = DetectabilityModel(num_units=num_cells,
                                       num_clases=4)
        base_arch.load_weights(args.path_model)

        # partially loading pretrained weights (multiclass training)
        base_arch.build((None, 40))
        weights_list = base_arch.get_weights()
        weights_list[-1] = np.array([0., 0.], dtype=np.float32)
        weights_list[-2] = np.zeros((128, 2), dtype=np.float32)
        fine_tuned_model.set_weights(weights_list)

    elif args.task == 'Regression':
        fine_tuned_model = DetectabilityModel(num_units=num_cells,
                                              num_clases=1)

        fine_tuned_model.decoder.decoder_dense = tf.keras.layers.Dense(1, activation=None)
        fine_tuned_model.build((None, 40))

        base_arch = DetectabilityModel(num_units=num_cells,
                                       num_clases=4)
        base_arch.load_weights(args.path_model)

        # partially loading pretrained weights (multiclass training)
        base_arch.build((None, 40))
        weights_list = base_arch.get_weights()
        weights_list[-1] = np.array([0.], dtype=np.float32)
        weights_list[-2] = np.zeros((128, 1), dtype=np.float32)
        fine_tuned_model.set_weights(weights_list)

    else:
        raise Exception('Task type not supported')

    max_pep_length = 40

    print('Initialising dataset')
    ## Data init
    fine_tune_data = DetectabilityDataset(data_source=args.path_dataset_train,
                                              val_data_source=args.path_dataset_val,
                                              data_format='csv',
                                              max_seq_len=max_pep_length,
                                              label_column="Classes",
                                              sequence_column="Sequences",
                                              dataset_columns_to_keep=["Proteins"],
                                              batch_size=args.batch_size,
                                              with_termini=False,
                                              alphabet=aa_to_int_dict)

    # compile the model  with the optimizer and the metrics we want to use.
    callback_FT = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   mode='min',
                                                   verbose=1,
                                                   patience=5)

    model_checkpoint_FT = tf.keras.callbacks.ModelCheckpoint(filepath=args.path_saved_model,
                                                             monitor='val_loss',
                                                             mode='min',
                                                             verbose=1,
                                                             save_best_only=True,
                                                             save_weights_only=True)
    opti = tf.keras.optimizers.legacy.Adagrad()

    if args.task == 'Multi'or 'Binary':
        fine_tuned_model.compile(optimizer=opti,
                                 loss='SparseCategoricalCrossentropy',
                                 metrics='sparse_categorical_accuracy')

    elif args.task == 'Regression':
        fine_tuned_model.compile(optimizer=opti,
                                 loss='MeanSquaredError',
                                 metrics='RootMeanSquaredError')

    else:
        raise Exception('Task type not supported')



    history_fine_tuned = fine_tuned_model.fit(fine_tune_data.tensor_train_data,
                                              validation_data=fine_tune_data.tensor_val_data,
                                              epochs=args.epochs,
                                              callbacks=[callback_FT, model_checkpoint_FT])

    ## Loading best model weights

    # model_save_path_FT = 'output/weights/new_fine_tuned_model/fine_tuned_model_weights_detectability_combined' #model fined tuned on ISA data
    # model_save_path_FT = 'pretrained_model/original_detectability_fine_tuned_model_FINAL' #base model

    fine_tuned_model.load_weights(args.path_saved_model)

    predictions_FT = fine_tuned_model.predict(fine_tune_data.tensor_val_data)

    # access val dataset and get the Classes column
    test_targets_FT = fine_tune_data["val"]["Classes"]

    # The dataframe needed for the report

    test_data_df_FT = pd.DataFrame(
        {
            "Sequences": fine_tune_data["val"]["_parsed_sequence"],  # get the raw parsed sequences
            "Classes": test_targets_FT,  # get the test targets from above
            "Proteins": fine_tune_data["val"]["Proteins"]  # get the Proteins column from the dataset object
        }
    )

    test_data_df_FT.Sequences = test_data_df_FT.Sequences.apply(lambda x: "".join(x))

    # Since the detectabiliy report expects the true labels in one-hot encoded format, we expand them here.

    num_classes = np.max(test_targets_FT) + 1
    test_targets_FT_one_hot = np.eye(num_classes)[test_targets_FT]

    report_FT = DetectabilityReport(test_targets_FT_one_hot,
                                    predictions_FT,
                                    test_data_df_FT,
                                    output_path=args.report_path,
                                    history=history_fine_tuned,
                                    rank_by_prot=True,
                                    threshold=None,
                                    name_of_dataset=args.path_dataset_train,
                                    name_of_model=args.model_name)

    report_FT.generate_report()