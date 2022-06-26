""" Compute averages from the chosen 4 configuration in the top 2 sigma confidence interval.
    Then plot the 4-run-averaged f1-score plot on the val set for the best of the 4 configurations
"""
from custom_plotting import plot_f1_score_comparison_quick, load_eval_csv_create_avg

# # # # # # # # # # # # # # # # #
# # # Cascade-RCNN average configs
list_eval_full_path = ["Cascade-RCNN_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.001_warmup_None_momentum_0.9_L2_3e-06_run_1/evals_f1_score/Cascade-RCNN_aug_True/lr_0.001_warmup_None_momentum_0.9_L2_3e-06_run_1/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
                       "Cascade-RCNN_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.001_warmup_None_momentum_0.9_L2_3e-06_run_2/evals_f1_score/Cascade-RCNN_aug_True/lr_0.001_warmup_None_momentum_0.9_L2_3e-06_run_2/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
                       "Cascade-RCNN_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.001_warmup_None_momentum_0.9_L2_3e-06_run_3/evals_f1_score/Cascade-RCNN_aug_True/lr_0.001_warmup_None_momentum_0.9_L2_3e-06_run_3/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
                       "Cascade-RCNN_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.001_warmup_None_momentum_0.9_L2_3e-06_run_4/evals_f1_score/Cascade-RCNN_aug_True/lr_0.001_warmup_None_momentum_0.9_L2_3e-06_run_4/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv"
                       ]
eval_csvs_config_1 = list_eval_full_path

list_eval_full_path = ["Cascade-RCNN_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.01_warmup_None_momentum_0.3_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_True/lr_0.01_warmup_None_momentum_0.3_L2_0.0003_run_1/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
                       "Cascade-RCNN_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.01_warmup_None_momentum_0.3_L2_0.0003_run_2/evals_f1_score/Cascade-RCNN_aug_True/lr_0.01_warmup_None_momentum_0.3_L2_0.0003_run_2/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
                       "Cascade-RCNN_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.01_warmup_None_momentum_0.3_L2_0.0003_run_3/evals_f1_score/Cascade-RCNN_aug_True/lr_0.01_warmup_None_momentum_0.3_L2_0.0003_run_3/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
                       "Cascade-RCNN_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.01_warmup_None_momentum_0.3_L2_0.0003_run_4/evals_f1_score/Cascade-RCNN_aug_True/lr_0.01_warmup_None_momentum_0.3_L2_0.0003_run_4/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv"
                       ]
eval_csvs_config_2 = list_eval_full_path

list_eval_full_path = ["Cascade-RCNN_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.001_warmup_None_momentum_0.3_L2_3e-06_run_1/evals_f1_score/Cascade-RCNN_aug_True/lr_0.001_warmup_None_momentum_0.3_L2_3e-06_run_1/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
                       "Cascade-RCNN_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.001_warmup_None_momentum_0.3_L2_3e-06_run_2/evals_f1_score/Cascade-RCNN_aug_True/lr_0.001_warmup_None_momentum_0.3_L2_3e-06_run_2/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
                       "Cascade-RCNN_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.001_warmup_None_momentum_0.3_L2_3e-06_run_3/evals_f1_score/Cascade-RCNN_aug_True/lr_0.001_warmup_None_momentum_0.3_L2_3e-06_run_3/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
                       "Cascade-RCNN_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.001_warmup_None_momentum_0.3_L2_3e-06_run_4/evals_f1_score/Cascade-RCNN_aug_True/lr_0.001_warmup_None_momentum_0.3_L2_3e-06_run_4/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv"
                       ]
eval_csvs_config_3 = list_eval_full_path

list_eval_full_path = ["Cascade-RCNN_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
                       "Cascade-RCNN_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_2/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_2/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
                       "Cascade-RCNN_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_3/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_3/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
                       "Cascade-RCNN_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_4/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_4/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv"
                       ]
eval_csvs_config_4 = list_eval_full_path

list_input_mode = ["Val"]
model_name = "Cascade R-CNN"
mode = "single"
gt_vers = ["regular"]
output = "Cascade-RCNN_Plot_Analysis/Plots"

output_main_path = "Cascade-RCNN_Plot_Analysis/02_Test_Grid_Search_Part_2/config_1_"
_, _, output_csv_mean_config_1, output_csv_std_config_1 = load_eval_csv_create_avg(eval_csvs=eval_csvs_config_1,
                                                                                   output_main_path=output_main_path)

output_main_path = "Cascade-RCNN_Plot_Analysis/02_Test_Grid_Search_Part_2/config_2_"
_, _, output_csv_mean_config_2, output_csv_std_config_2 = load_eval_csv_create_avg(eval_csvs=eval_csvs_config_2,
                                                                                   output_main_path=output_main_path)

output_main_path = "Cascade-RCNN_Plot_Analysis/02_Test_Grid_Search_Part_2/config_3_"
_, _, output_csv_mean_config_3, output_csv_std_config_3 = load_eval_csv_create_avg(eval_csvs=eval_csvs_config_3,
                                                                                   output_main_path=output_main_path)

output_main_path = "Cascade-RCNN_Plot_Analysis/02_Test_Grid_Search_Part_2/config_4_"
_, _, output_csv_mean_config_4, output_csv_std_config_4 = load_eval_csv_create_avg(eval_csvs=eval_csvs_config_4,
                                                                                   output_main_path=output_main_path)

legend = ["config 1", "config 2", "config 3", "config 4"]
plot_f1_score_comparison_quick([output_csv_mean_config_1, output_csv_mean_config_2, output_csv_mean_config_3,
                                output_csv_mean_config_4], list_input_mode, model_name, mode,
                               gt_vers, legend,
                               list_std_full_path=None,
                               # list_std_full_path=[output_csv_std_config_1, output_csv_std_config_2,
                               #                     output_csv_std_config_3, output_csv_std_config_4],
                               output=output,
                               smooth=True, smoothing_size=10, ylim=[0.6, 0.84],
                               plot_name_postfix="Cascade_RCNN_best_4_AVG_smooth_10")


# # # # # # # # # # # # # # # # #
# # # VFNet average configs
list_eval_full_path = ["VFNet_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.001_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/VFNet_aug_True/lr_0.001_warmup_None_momentum_0.9_L2_0.0003_run_1/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "VFNet_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.001_warmup_None_momentum_0.9_L2_0.0003_run_2/evals_f1_score/VFNet_aug_True/lr_0.001_warmup_None_momentum_0.9_L2_0.0003_run_2/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "VFNet_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.001_warmup_None_momentum_0.9_L2_0.0003_run_3/evals_f1_score/VFNet_aug_True/lr_0.001_warmup_None_momentum_0.9_L2_0.0003_run_3/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "VFNet_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.001_warmup_None_momentum_0.9_L2_0.0003_run_4/evals_f1_score/VFNet_aug_True/lr_0.001_warmup_None_momentum_0.9_L2_0.0003_run_4/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv"
                       ]
eval_csvs_config_1 = list_eval_full_path

list_eval_full_path = ["VFNet_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.0001_warmup_None_momentum_0.9_L2_3e-06_run_1/evals_f1_score/VFNet_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_3e-06_run_1/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "VFNet_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.0001_warmup_None_momentum_0.9_L2_3e-06_run_2/evals_f1_score/VFNet_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_3e-06_run_2/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "VFNet_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.0001_warmup_None_momentum_0.9_L2_3e-06_run_3/evals_f1_score/VFNet_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_3e-06_run_3/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "VFNet_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.0001_warmup_None_momentum_0.9_L2_3e-06_run_4/evals_f1_score/VFNet_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_3e-06_run_4/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv"
                       ]
eval_csvs_config_2 = list_eval_full_path

list_eval_full_path = ["VFNet_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.001_warmup_None_momentum_0.0_L2_0.03_run_1/evals_f1_score/VFNet_aug_True/lr_0.001_warmup_None_momentum_0.0_L2_0.03_run_1/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "VFNet_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.001_warmup_None_momentum_0.0_L2_0.03_run_2/evals_f1_score/VFNet_aug_True/lr_0.001_warmup_None_momentum_0.0_L2_0.03_run_2/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "VFNet_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.001_warmup_None_momentum_0.0_L2_0.03_run_3/evals_f1_score/VFNet_aug_True/lr_0.001_warmup_None_momentum_0.0_L2_0.03_run_3/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "VFNet_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.001_warmup_None_momentum_0.0_L2_0.03_run_4/evals_f1_score/VFNet_aug_True/lr_0.001_warmup_None_momentum_0.0_L2_0.03_run_4/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv"
                       ]
eval_csvs_config_3 = list_eval_full_path

list_eval_full_path = ["VFNet_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.0001_warmup_None_momentum_0.3_L2_0.0003_run_1/evals_f1_score/VFNet_aug_True/lr_0.0001_warmup_None_momentum_0.3_L2_0.0003_run_1/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "VFNet_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.0001_warmup_None_momentum_0.3_L2_0.0003_run_2/evals_f1_score/VFNet_aug_True/lr_0.0001_warmup_None_momentum_0.3_L2_0.0003_run_2/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "VFNet_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.0001_warmup_None_momentum_0.3_L2_0.0003_run_3/evals_f1_score/VFNet_aug_True/lr_0.0001_warmup_None_momentum_0.3_L2_0.0003_run_3/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "VFNet_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.0001_warmup_None_momentum_0.3_L2_0.0003_run_4/evals_f1_score/VFNet_aug_True/lr_0.0001_warmup_None_momentum_0.3_L2_0.0003_run_4/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv"
                       ]
eval_csvs_config_4 = list_eval_full_path

list_input_mode = ["Val"]
model_name = "VFNet"
mode = "single"
gt_vers = ["regular"]
output = "VFNet_Plot_Analysis/Plots"

output_main_path = "VFNet_Plot_Analysis/02_Test_Grid_Search_Part_2/config_1_"
_, _, output_csv_mean_config_1, output_csv_std_config_1 = load_eval_csv_create_avg(eval_csvs=eval_csvs_config_1,
                                                                                   output_main_path=output_main_path)
output_main_path = "VFNet_Plot_Analysis/02_Test_Grid_Search_Part_2/config_2_"
_, _, output_csv_mean_config_2, output_csv_std_config_2 = load_eval_csv_create_avg(eval_csvs=eval_csvs_config_2,
                                                                                   output_main_path=output_main_path)
output_main_path = "VFNet_Plot_Analysis/02_Test_Grid_Search_Part_2/config_3_"
_, _, output_csv_mean_config_3, output_csv_std_config_3 = load_eval_csv_create_avg(eval_csvs=eval_csvs_config_3,
                                                                                   output_main_path=output_main_path)
output_main_path = "VFNet_Plot_Analysis/02_Test_Grid_Search_Part_2/config_4_"
_, _, output_csv_mean_config_4, output_csv_std_config_4 = load_eval_csv_create_avg(eval_csvs=eval_csvs_config_4,
                                                                                   output_main_path=output_main_path)

legend = ["config 1", "config 2", "config 3", "config 4"]
plot_f1_score_comparison_quick([output_csv_mean_config_1, output_csv_mean_config_2, output_csv_mean_config_3,
                                output_csv_mean_config_4], list_input_mode, model_name, mode,
                               gt_vers, legend,
                               # list_std_full_path=None,
                               list_std_full_path=[output_csv_std_config_1, output_csv_std_config_2,
                                                   output_csv_std_config_3, output_csv_std_config_4],
                               output=output,
                               smooth=False, smoothing_size=10, ylim=[0.6, 0.84],
                               plot_name_postfix="VFNet_best_4_AVG_STD")


# # # # # # # # # # # # # # # # #
# # DETR average configs
list_eval_full_path = ["Def_DETR_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.0001_warmup_None_dropout_0.1_momentum_0.6_L2_3e-06_run_1/evals_f1_score/Def_DETR_aug_True/lr_0.0001_warmup_None_dropout_0.1_momentum_0.6_L2_3e-06_run_1/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.0001_warmup_None_dropout_0.1_momentum_0.6_L2_3e-06_run_2/evals_f1_score/Def_DETR_aug_True/lr_0.0001_warmup_None_dropout_0.1_momentum_0.6_L2_3e-06_run_2/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.0001_warmup_None_dropout_0.1_momentum_0.6_L2_3e-06_run_3/evals_f1_score/Def_DETR_aug_True/lr_0.0001_warmup_None_dropout_0.1_momentum_0.6_L2_3e-06_run_3/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.0001_warmup_None_dropout_0.1_momentum_0.6_L2_3e-06_run_4/evals_f1_score/Def_DETR_aug_True/lr_0.0001_warmup_None_dropout_0.1_momentum_0.6_L2_3e-06_run_4/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv"
                       ]
eval_csvs_config_1 = list_eval_full_path

list_eval_full_path = ["Def_DETR_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.001_warmup_None_dropout_0.1_momentum_0.3_L2_3e-06_run_1/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.1_momentum_0.3_L2_3e-06_run_1/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.001_warmup_None_dropout_0.1_momentum_0.3_L2_3e-06_run_2/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.1_momentum_0.3_L2_3e-06_run_2/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.001_warmup_None_dropout_0.1_momentum_0.3_L2_3e-06_run_3/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.1_momentum_0.3_L2_3e-06_run_3/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.001_warmup_None_dropout_0.1_momentum_0.3_L2_3e-06_run_4/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.1_momentum_0.3_L2_3e-06_run_4/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv"
                       ]
eval_csvs_config_2 = list_eval_full_path

list_eval_full_path = ["Def_DETR_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.03_run_1/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.03_run_1/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.03_run_2/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.03_run_2/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.03_run_3/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.03_run_3/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.03_run_4/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.03_run_4/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv"
                       ]
eval_csvs_config_3 = list_eval_full_path

list_eval_full_path = ["Def_DETR_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.001_warmup_None_dropout_0.1_momentum_0.0_L2_0.03_run_1/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.1_momentum_0.0_L2_0.03_run_1/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.001_warmup_None_dropout_0.1_momentum_0.0_L2_0.03_run_2/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.1_momentum_0.0_L2_0.03_run_2/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.001_warmup_None_dropout_0.1_momentum_0.0_L2_0.03_run_3/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.1_momentum_0.0_L2_0.03_run_3/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.001_warmup_None_dropout_0.1_momentum_0.0_L2_0.03_run_4/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.1_momentum_0.0_L2_0.03_run_4/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv"
                       ]
eval_csvs_config_4 = list_eval_full_path

list_input_mode = ["Val"]
model_name = "Def DETR"
mode = "single"
gt_vers = ["regular"]
output = "Def_DETR_Plot_Analysis/Plots"

output_main_path = "Def_DETR_Plot_Analysis/02_Test_Grid_Search_Part_2/config_1_"
_, _, output_csv_mean_config_1, output_csv_std_config_1 = load_eval_csv_create_avg(eval_csvs=eval_csvs_config_1,
                                                                                   output_main_path=output_main_path)
output_main_path = "Def_DETR_Plot_Analysis/02_Test_Grid_Search_Part_2/config_2_"
_, _, output_csv_mean_config_2, output_csv_std_config_2 = load_eval_csv_create_avg(eval_csvs=eval_csvs_config_2,
                                                                                   output_main_path=output_main_path)
output_main_path = "Def_DETR_Plot_Analysis/02_Test_Grid_Search_Part_2/config_3_"
_, _, output_csv_mean_config_3, output_csv_std_config_3 = load_eval_csv_create_avg(eval_csvs=eval_csvs_config_3,
                                                                                   output_main_path=output_main_path)
output_main_path = "Def_DETR_Plot_Analysis/02_Test_Grid_Search_Part_2/config_4_"
_, _, output_csv_mean_config_4, output_csv_std_config_4 = load_eval_csv_create_avg(eval_csvs=eval_csvs_config_4,
                                                                                   output_main_path=output_main_path)

legend = ["config 1", "config 2", "config 3", "config 4"]
plot_f1_score_comparison_quick([output_csv_mean_config_1, output_csv_mean_config_2, output_csv_mean_config_3,
                                output_csv_mean_config_4], list_input_mode, model_name, mode,
                               gt_vers, legend,
                               list_std_full_path=None,
                               # list_std_full_path=[output_csv_std_config_1, output_csv_std_config_2,
                               #                     output_csv_std_config_3, output_csv_std_config_4],
                               output=output,
                               smooth=True, smoothing_size=10, ylim=[0.6, 0.84],
                               plot_name_postfix="Def_DETR_best_4_AVG_smooth_10")
