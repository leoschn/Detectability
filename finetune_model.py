from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import initializers
from os.path import join, exists
from os import makedirs
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve
from dlomix.models.detectability import DetectabilityModel
from tensorflow.keras.callbacks import TerminateOnNaN
from dlomix.constants import CLASSES_LABELS, aa_to_int_dict
from dlomix.data import DetectabilityDataset
from dlomix.reports.DetectabilityReport import DetectabilityReport
from config import load_args_finetune
import os



def plot_and_save_metrics(history, base_path):
    history_dict = history.history
    metrics = history_dict.keys()
    metrics = filter(lambda x: not x.startswith(tuple(["val_", "_"])), metrics)

    if not exists(base_path):
        makedirs(base_path)

    for metric_name in metrics:
        plt.plot(history_dict[metric_name])
        plt.plot(history_dict["val_" + metric_name])
        plt.title(metric_name, fontsize=10)  # Modified Original plt.title(metric_name)
        plt.ylabel(metric_name)
        plt.xlabel("epoch")
        plt.legend(["train", "val"], loc="best")
        save_path = join(base_path, metric_name)
        plt.savefig(
            save_path, bbox_inches="tight", dpi=90
        )  # Modification Original plt.savefig(save_path)
        plt.close()

def plot_confusion_matrix_binary(df, base_path):
    conf_matrix = confusion_matrix(
        df["Classes"],
        df["Predicted Classes"],
    )

    if not exists(base_path):
        makedirs(base_path)

    conf_matrix_disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_matrix, display_labels=["Non-Flyer", "Flyer"]
    )
    fig, ax = plt.subplots()
    conf_matrix_disp.plot(xticks_rotation=45, ax=ax)
    plt.title("Confusion Matrix (Binary Classification)", y=1.04, fontsize=11)
    save_path = join(base_path, "confusion_matrix_binary"
    )
    plt.savefig(save_path, bbox_inches="tight", dpi=80)
    plt.close()

def plot_roc_binary(df,base_path):
    fpr, tpr, thresholds = roc_curve(
        np.array(df["Classes"]),
        np.array(df["Predictions"]),
    )
    AUC_score = auc(fpr, tpr)

    # create ROC curve

    plt.plot(fpr, tpr, label="ROC curve of (area = {})".format(AUC_score))
    plt.title(
        "Receiver operating characteristic curve (Binary classification)",
        y=1.04,
        fontsize=10,
    )

    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    save_path = join(
        base_path, "ROC_curve_binary_classification"
    )

    plt.savefig(save_path, bbox_inches="tight", dpi=90)
    plt.close()

def plot_scatter(df,base_path):
    plt.scatter(
        df["Classes"],
        df["Predictions"],s=0.25,alpha=0.3
    )
    plt.title(
        "Scatter plot of detectability (log of relative intensity)",
        y=1.04,
        fontsize=10,
    )
    plt.ylabel("Predicted detectability")
    plt.xlabel("True detectability")
    save_path = join(
        base_path, "scatter_plots"
    )
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    args = load_args_finetune()
    print(args)

    num_cells = 64
    total_num_classes = len(CLASSES_LABELS)


    if args.task =='Multi_class':
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
                                              num_clases=4)

        fine_tuned_model.load_weights(args.path_model)
        fine_tuned_model.decoder.decoder_dense = tf.keras.layers.Dense(1, activation=None,kernel_initializer=initializers.GlorotNormal(seed=None),
    bias_initializer=initializers.Zeros())
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
                                              label_column="Label MaxLFQ",
                                              sequence_column="Sequences",
                                              dataset_columns_to_keep=['Protein.Ids'],
                                              batch_size=args.batch_size,
                                              with_termini=False,
                                              alphabet=aa_to_int_dict)


    model_checkpoint_FT = tf.keras.callbacks.ModelCheckpoint(filepath=args.save_model_path,
                                                             monitor='val_loss',
                                                             mode='min',
                                                             verbose=1,
                                                             save_best_only=False,
                                                             save_weights_only=True)
    opti = tf.keras.optimizers.legacy.Adagrad()

    if args.task == 'Multi_class'or args.task=='Binary':
        fine_tuned_model.compile(optimizer=opti,
                                 loss='SparseCategoricalCrossentropy',
                                 metrics='sparse_categorical_accuracy')

    elif args.task == 'Regression':
        fine_tuned_model.compile(optimizer=opti,
                                 loss='MeanSquaredError',
                                 metrics='MeanSquaredError')

    else:
        raise Exception('Task type not supported')

    history_fine_tuned = fine_tuned_model.fit(fine_tune_data.tensor_train_data,
                                              validation_data=fine_tune_data.tensor_val_data,
                                              epochs=args.epochs,
                                              callbacks=[model_checkpoint_FT])

    # Loading best model weights

    fine_tuned_model.load_weights(args.save_model_path)

    predictions_FT = fine_tuned_model.predict(fine_tune_data.tensor_val_data)

    # access val dataset and get the Classes column
    test_targets_FT = fine_tune_data["val"]["Label MaxLFQ"]

    # The dataframe needed for the report

    test_data_df_FT = pd.DataFrame(
        {
            "Sequences": fine_tune_data["val"]["_parsed_sequence"],  # get the raw parsed sequences
            "Classes": test_targets_FT,  # get the test targets from above
            "Proteins": fine_tune_data["val"]["Protein.Ids"]  # get the Proteins column from the dataset object
        }
    )

    test_data_df_FT.Sequences = test_data_df_FT.Sequences.apply(lambda x: "".join(x))

    # Since the detectabiliy report expects the true labels in one-hot encoded format, we expand them here.

    if not(os.path.exists(args.report_path)):
        os.makedirs(args.report_path)

    if args.task=='Multi_class' :
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

    elif args.task=='Binary':
        df = pd.DataFrame(data={'Classes':test_targets_FT,'Predictions':predictions_FT[:,1],'Predicted Classes':np.argmax(predictions_FT,axis=1)})
        plot_confusion_matrix_binary(df,args.report_path)
        plot_roc_binary(df,args.report_path)
        plot_and_save_metrics(history_fine_tuned,args.report_path)

    elif args.task=='Regression':
        df = pd.DataFrame(data={'Classes': test_targets_FT, 'Predictions': predictions_FT[:, 0]})
        plot_scatter(df, args.report_path)
        plot_and_save_metrics(history_fine_tuned, args.report_path)

