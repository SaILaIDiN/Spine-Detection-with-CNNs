""" Compute averages for three dropout cases of Def-DETR. """
from custom_plotting import plot_f1_score_comparison_quick, load_eval_csv_create_avg


# # # # # # # # # # # # # # # # #
# # DETR average configs
list_eval_full_path = ["Def_DETR_Plot_Analysis/03_Test_Dropout_Part_2/lr_0.001_warmup_None_dropout_0.0_momentum_0.3_L2_3e-06_run_1/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.0_momentum_0.3_L2_3e-06_run_1/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/03_Test_Dropout_Part_2/lr_0.001_warmup_None_dropout_0.0_momentum_0.3_L2_3e-06_run_2/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.0_momentum_0.3_L2_3e-06_run_2/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/03_Test_Dropout_Part_2/lr_0.001_warmup_None_dropout_0.0_momentum_0.3_L2_3e-06_run_3/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.0_momentum_0.3_L2_3e-06_run_3/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/03_Test_Dropout_Part_2/lr_0.001_warmup_None_dropout_0.0_momentum_0.3_L2_3e-06_run_4/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.0_momentum_0.3_L2_3e-06_run_4/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv"
                       ]
eval_csvs_config_0 = list_eval_full_path


list_eval_full_path = ["Def_DETR_Plot_Analysis/03_Test_Dropout/lr_0.001_warmup_None_dropout_0.1_momentum_0.3_L2_3e-06_run_1/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.1_momentum_0.3_L2_3e-06_run_1/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/03_Test_Dropout/lr_0.001_warmup_None_dropout_0.1_momentum_0.3_L2_3e-06_run_2/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.1_momentum_0.3_L2_3e-06_run_2/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/03_Test_Dropout/lr_0.001_warmup_None_dropout_0.1_momentum_0.3_L2_3e-06_run_3/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.1_momentum_0.3_L2_3e-06_run_3/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/03_Test_Dropout/lr_0.001_warmup_None_dropout_0.1_momentum_0.3_L2_3e-06_run_4/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.1_momentum_0.3_L2_3e-06_run_4/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv"
                       ]
eval_csvs_config_1 = list_eval_full_path

list_eval_full_path = ["Def_DETR_Plot_Analysis/03_Test_Dropout/lr_0.001_warmup_None_dropout_0.3_momentum_0.3_L2_3e-06_run_1/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.3_momentum_0.3_L2_3e-06_run_1/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/03_Test_Dropout/lr_0.001_warmup_None_dropout_0.3_momentum_0.3_L2_3e-06_run_2/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.3_momentum_0.3_L2_3e-06_run_2/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/03_Test_Dropout/lr_0.001_warmup_None_dropout_0.3_momentum_0.3_L2_3e-06_run_3/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.3_momentum_0.3_L2_3e-06_run_3/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/03_Test_Dropout/lr_0.001_warmup_None_dropout_0.3_momentum_0.3_L2_3e-06_run_4/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.3_momentum_0.3_L2_3e-06_run_4/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv"
                       ]
eval_csvs_config_2 = list_eval_full_path

list_eval_full_path = ["Def_DETR_Plot_Analysis/03_Test_Dropout/lr_0.001_warmup_None_dropout_0.5_momentum_0.3_L2_3e-06_run_1/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.5_momentum_0.3_L2_3e-06_run_1/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/03_Test_Dropout/lr_0.001_warmup_None_dropout_0.5_momentum_0.3_L2_3e-06_run_2/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.5_momentum_0.3_L2_3e-06_run_2/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/03_Test_Dropout/lr_0.001_warmup_None_dropout_0.5_momentum_0.3_L2_3e-06_run_3/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.5_momentum_0.3_L2_3e-06_run_3/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/03_Test_Dropout/lr_0.001_warmup_None_dropout_0.5_momentum_0.3_L2_3e-06_run_4/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.5_momentum_0.3_L2_3e-06_run_4/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv"
                       ]
eval_csvs_config_3 = list_eval_full_path

list_input_mode = ["Val", "Val", "Val", "Val"]
model_name = "Def DETR"
mode = "single"
gt_vers = ["regular", "regular", "regular", "regular"]
output = "Def_DETR_Plot_Analysis/Plots"

output_main_path = "Def_DETR_Plot_Analysis/03_Test_Dropout_Part_2/config_0_"
_, _, output_csv_mean_config_0, output_csv_std_config_0 = load_eval_csv_create_avg(eval_csvs=eval_csvs_config_0,
                                                                                   output_main_path=output_main_path)

output_main_path = "Def_DETR_Plot_Analysis/03_Test_Dropout/config_1_"
_, _, output_csv_mean_config_1, output_csv_std_config_1 = load_eval_csv_create_avg(eval_csvs=eval_csvs_config_1,
                                                                                   output_main_path=output_main_path)
output_main_path = "Def_DETR_Plot_Analysis/03_Test_Dropout/config_2_"
_, _, output_csv_mean_config_2, output_csv_std_config_2 = load_eval_csv_create_avg(eval_csvs=eval_csvs_config_2,
                                                                                   output_main_path=output_main_path)
output_main_path = "Def_DETR_Plot_Analysis/03_Test_Dropout/config_3_"
_, _, output_csv_mean_config_3, output_csv_std_config_3 = load_eval_csv_create_avg(eval_csvs=eval_csvs_config_3,
                                                                                   output_main_path=output_main_path)


legend = ["dropout 0.0", "dropout 0.1", "dropout 0.3", "dropout 0.5"]
plot_f1_score_comparison_quick([output_csv_mean_config_0, output_csv_mean_config_1, output_csv_mean_config_2,
                                output_csv_mean_config_3],
                               list_input_mode, model_name, mode,
                               gt_vers, legend,
                               # list_std_full_path=None,
                               list_std_full_path=[output_csv_std_config_0, output_csv_std_config_1,
                                                   output_csv_std_config_2, output_csv_std_config_3],
                               output=output,
                               smooth=True, smoothing_size=10, ylim=[0.6, 0.84],
                               plot_name_postfix="Def_DETR_dropout_AVG_STD_smooth_10")
