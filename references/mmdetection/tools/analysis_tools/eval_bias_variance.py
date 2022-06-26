""" Analyze bias and variance of the final configuration of each network.
    Plot average f1-scores of train, val and test in the same figure. Three figures in total, one per architecture.
"""
from custom_plotting import plot_f1_score_comparison_quick, load_eval_csv_create_avg


# # # # # # # # # # # # # # # # #
# # # Cascade-RCNN average configs
list_eval_full_path = ["Cascade-RCNN_Plot_Analysis/02_Test_Grid_Search_Part_3/lr_0.01_warmup_None_momentum_0.3_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_True/lr_0.01_warmup_None_momentum_0.3_L2_0.0003_run_1/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Train_eval.csv",
                       "Cascade-RCNN_Plot_Analysis/02_Test_Grid_Search_Part_3/lr_0.01_warmup_None_momentum_0.3_L2_0.0003_run_2/evals_f1_score/Cascade-RCNN_aug_True/lr_0.01_warmup_None_momentum_0.3_L2_0.0003_run_2/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Train_eval.csv",
                       "Cascade-RCNN_Plot_Analysis/02_Test_Grid_Search_Part_3/lr_0.01_warmup_None_momentum_0.3_L2_0.0003_run_3/evals_f1_score/Cascade-RCNN_aug_True/lr_0.01_warmup_None_momentum_0.3_L2_0.0003_run_3/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Train_eval.csv",
                       "Cascade-RCNN_Plot_Analysis/02_Test_Grid_Search_Part_3/lr_0.01_warmup_None_momentum_0.3_L2_0.0003_run_4/evals_f1_score/Cascade-RCNN_aug_True/lr_0.01_warmup_None_momentum_0.3_L2_0.0003_run_4/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Train_eval.csv"
                       ]
eval_csvs_config_cascade_Train = list_eval_full_path

list_eval_full_path = ["Cascade-RCNN_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.01_warmup_None_momentum_0.3_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_True/lr_0.01_warmup_None_momentum_0.3_L2_0.0003_run_1/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
                       "Cascade-RCNN_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.01_warmup_None_momentum_0.3_L2_0.0003_run_2/evals_f1_score/Cascade-RCNN_aug_True/lr_0.01_warmup_None_momentum_0.3_L2_0.0003_run_2/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
                       "Cascade-RCNN_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.01_warmup_None_momentum_0.3_L2_0.0003_run_3/evals_f1_score/Cascade-RCNN_aug_True/lr_0.01_warmup_None_momentum_0.3_L2_0.0003_run_3/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
                       "Cascade-RCNN_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.01_warmup_None_momentum_0.3_L2_0.0003_run_4/evals_f1_score/Cascade-RCNN_aug_True/lr_0.01_warmup_None_momentum_0.3_L2_0.0003_run_4/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv"
                       ]
eval_csvs_config_cascade_Val = list_eval_full_path

list_eval_full_path = ["Cascade-RCNN_Plot_Analysis/02_Test_Grid_Search_Part_3/lr_0.01_warmup_None_momentum_0.3_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_True/lr_0.01_warmup_None_momentum_0.3_L2_0.0003_run_1/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Test_eval.csv",
                       "Cascade-RCNN_Plot_Analysis/02_Test_Grid_Search_Part_3/lr_0.01_warmup_None_momentum_0.3_L2_0.0003_run_2/evals_f1_score/Cascade-RCNN_aug_True/lr_0.01_warmup_None_momentum_0.3_L2_0.0003_run_2/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Test_eval.csv",
                       "Cascade-RCNN_Plot_Analysis/02_Test_Grid_Search_Part_3/lr_0.01_warmup_None_momentum_0.3_L2_0.0003_run_3/evals_f1_score/Cascade-RCNN_aug_True/lr_0.01_warmup_None_momentum_0.3_L2_0.0003_run_3/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Test_eval.csv",
                       "Cascade-RCNN_Plot_Analysis/02_Test_Grid_Search_Part_3/lr_0.01_warmup_None_momentum_0.3_L2_0.0003_run_4/evals_f1_score/Cascade-RCNN_aug_True/lr_0.01_warmup_None_momentum_0.3_L2_0.0003_run_4/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Test_eval.csv"
                       ]
eval_csvs_config_cascade_Test = list_eval_full_path

list_input_mode = ["Train", "Val", "Test"]
model_name = "Cascade R-CNN"
mode = "single"
gt_vers = ["regular", "regular", "maj"]
output = "Cascade-RCNN_Plot_Analysis/Plots"

output_main_path = "Cascade-RCNN_Plot_Analysis/02_Test_Grid_Search_Part_3/config_Train_"
_, _, output_csv_mean_config_cascade_Train, output_csv_std_config_cascade_Train = load_eval_csv_create_avg(
    eval_csvs=eval_csvs_config_cascade_Train, output_main_path=output_main_path)

output_main_path = "Cascade-RCNN_Plot_Analysis/02_Test_Grid_Search_Part_2/config_Val_"
_, _, output_csv_mean_config_cascade_Val, output_csv_std_config_cascade_Val = load_eval_csv_create_avg(
    eval_csvs=eval_csvs_config_cascade_Val, output_main_path=output_main_path)

output_main_path = "Cascade-RCNN_Plot_Analysis/02_Test_Grid_Search_Part_3/config_Test_"
_, _, output_csv_mean_config_cascade_Test, output_csv_std_config_cascade_Test = load_eval_csv_create_avg(
    eval_csvs=eval_csvs_config_cascade_Test, output_main_path=output_main_path)

legend = ["Train", "Val", "Test"]
plot_f1_score_comparison_quick([output_csv_mean_config_cascade_Train, output_csv_mean_config_cascade_Val,
                                output_csv_mean_config_cascade_Test],
                               list_input_mode, model_name, mode,
                               gt_vers, legend,
                               # list_std_full_path=None,
                               list_std_full_path=[output_csv_std_config_cascade_Train,
                                                   output_csv_std_config_cascade_Val,
                                                   output_csv_std_config_cascade_Test],
                               output=output,
                               smooth=True, smoothing_size=10, ylim=[0.6, 0.90],
                               plot_name_postfix="Cascade_RCNN_bias_variance_AVG_smooth_10")


# # # # # # # # # # # # # # # # #
# # # VFNet average configs
list_eval_full_path = ["VFNet_Plot_Analysis/02_Test_Grid_Search_Part_3/lr_0.0001_warmup_None_momentum_0.9_L2_3e-06_run_1/evals_f1_score/VFNet_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_3e-06_run_1/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Train_eval.csv",
                       "VFNet_Plot_Analysis/02_Test_Grid_Search_Part_3/lr_0.0001_warmup_None_momentum_0.9_L2_3e-06_run_2/evals_f1_score/VFNet_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_3e-06_run_2/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Train_eval.csv",
                       "VFNet_Plot_Analysis/02_Test_Grid_Search_Part_3/lr_0.0001_warmup_None_momentum_0.9_L2_3e-06_run_3/evals_f1_score/VFNet_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_3e-06_run_3/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Train_eval.csv",
                       "VFNet_Plot_Analysis/02_Test_Grid_Search_Part_3/lr_0.0001_warmup_None_momentum_0.9_L2_3e-06_run_4/evals_f1_score/VFNet_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_3e-06_run_4/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Train_eval.csv"
                       ]
eval_csvs_config_vfnet_Train = list_eval_full_path

list_eval_full_path = ["VFNet_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.0001_warmup_None_momentum_0.9_L2_3e-06_run_1/evals_f1_score/VFNet_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_3e-06_run_1/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "VFNet_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.0001_warmup_None_momentum_0.9_L2_3e-06_run_2/evals_f1_score/VFNet_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_3e-06_run_2/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "VFNet_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.0001_warmup_None_momentum_0.9_L2_3e-06_run_3/evals_f1_score/VFNet_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_3e-06_run_3/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "VFNet_Plot_Analysis/02_Test_Grid_Search_Part_2/lr_0.0001_warmup_None_momentum_0.9_L2_3e-06_run_4/evals_f1_score/VFNet_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_3e-06_run_4/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv"
                       ]
eval_csvs_config_vfnet_Val = list_eval_full_path

list_eval_full_path = ["VFNet_Plot_Analysis/02_Test_Grid_Search_Part_3/lr_0.0001_warmup_None_momentum_0.9_L2_3e-06_run_1/evals_f1_score/VFNet_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_3e-06_run_1/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Test_eval.csv",
                       "VFNet_Plot_Analysis/02_Test_Grid_Search_Part_3/lr_0.0001_warmup_None_momentum_0.9_L2_3e-06_run_2/evals_f1_score/VFNet_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_3e-06_run_2/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Test_eval.csv",
                       "VFNet_Plot_Analysis/02_Test_Grid_Search_Part_3/lr_0.0001_warmup_None_momentum_0.9_L2_3e-06_run_3/evals_f1_score/VFNet_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_3e-06_run_3/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Test_eval.csv",
                       "VFNet_Plot_Analysis/02_Test_Grid_Search_Part_3/lr_0.0001_warmup_None_momentum_0.9_L2_3e-06_run_4/evals_f1_score/VFNet_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_3e-06_run_4/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Test_eval.csv"
                       ]
eval_csvs_config_vfnet_Test = list_eval_full_path

list_input_mode = ["Train", "Val", "Test"]
model_name = "VFNet"
mode = "single"
gt_vers = ["regular", "regular", "maj"]
output = "VFNet_Plot_Analysis/Plots"

output_main_path = "VFNet_Plot_Analysis/02_Test_Grid_Search_Part_2/config_Train"
_, _, output_csv_mean_config_vfnet_Train, output_csv_std_config_vfnet_Train = load_eval_csv_create_avg(
    eval_csvs=eval_csvs_config_vfnet_Train, output_main_path=output_main_path)

output_main_path = "VFNet_Plot_Analysis/02_Test_Grid_Search_Part_2/config_Val"
_, _, output_csv_mean_config_vfnet_Val, output_csv_std_config_vfnet_Val = load_eval_csv_create_avg(
    eval_csvs=eval_csvs_config_vfnet_Val, output_main_path=output_main_path)

output_main_path = "VFNet_Plot_Analysis/02_Test_Grid_Search_Part_2/config_Test"
_, _, output_csv_mean_config_vfnet_Test, output_csv_std_config_vfnet_Test = load_eval_csv_create_avg(
    eval_csvs=eval_csvs_config_vfnet_Test, output_main_path=output_main_path)

legend = ["Train", "Val", "Test"]
plot_f1_score_comparison_quick([output_csv_mean_config_vfnet_Train, output_csv_mean_config_vfnet_Val,
                                output_csv_mean_config_vfnet_Test],
                               list_input_mode, model_name, mode,
                               gt_vers, legend,
                               # list_std_full_path=None,
                               list_std_full_path=[output_csv_std_config_vfnet_Train, output_csv_std_config_vfnet_Val,
                                                   output_csv_std_config_vfnet_Test],
                               output=output,
                               smooth=True, smoothing_size=10, ylim=[0.6, 0.90],
                               plot_name_postfix="VFNet_bias_and_variance_AVG_smooth_10")


# # # # # # # # # # # # # # # # #
# # # Def-DETR average configs
list_eval_full_path = ["Def_DETR_Plot_Analysis/03_Test_Dropout_Part_4/lr_0.001_warmup_None_dropout_0.0_momentum_0.3_L2_3e-06_run_1/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.0_momentum_0.3_L2_3e-06_run_1/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Train_eval.csv",
                       "Def_DETR_Plot_Analysis/03_Test_Dropout_Part_4/lr_0.001_warmup_None_dropout_0.0_momentum_0.3_L2_3e-06_run_2/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.0_momentum_0.3_L2_3e-06_run_2/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Train_eval.csv",
                       "Def_DETR_Plot_Analysis/03_Test_Dropout_Part_4/lr_0.001_warmup_None_dropout_0.0_momentum_0.3_L2_3e-06_run_3/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.0_momentum_0.3_L2_3e-06_run_3/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Train_eval.csv",
                       "Def_DETR_Plot_Analysis/03_Test_Dropout_Part_4/lr_0.001_warmup_None_dropout_0.0_momentum_0.3_L2_3e-06_run_4/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.0_momentum_0.3_L2_3e-06_run_4/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Train_eval.csv"
                       ]
eval_csvs_config_detr_Train = list_eval_full_path

list_eval_full_path = ["Def_DETR_Plot_Analysis/03_Test_Dropout_Part_2/lr_0.001_warmup_None_dropout_0.0_momentum_0.3_L2_3e-06_run_1/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.0_momentum_0.3_L2_3e-06_run_1/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/03_Test_Dropout_Part_2/lr_0.001_warmup_None_dropout_0.0_momentum_0.3_L2_3e-06_run_2/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.0_momentum_0.3_L2_3e-06_run_2/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/03_Test_Dropout_Part_2/lr_0.001_warmup_None_dropout_0.0_momentum_0.3_L2_3e-06_run_3/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.0_momentum_0.3_L2_3e-06_run_3/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/03_Test_Dropout_Part_2/lr_0.001_warmup_None_dropout_0.0_momentum_0.3_L2_3e-06_run_4/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.0_momentum_0.3_L2_3e-06_run_4/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv"
                       ]
eval_csvs_config_detr_Val = list_eval_full_path

list_eval_full_path = ["Def_DETR_Plot_Analysis/03_Test_Dropout_Part_4/lr_0.001_warmup_None_dropout_0.0_momentum_0.3_L2_3e-06_run_1/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.0_momentum_0.3_L2_3e-06_run_1/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Test_eval.csv",
                       "Def_DETR_Plot_Analysis/03_Test_Dropout_Part_4/lr_0.001_warmup_None_dropout_0.0_momentum_0.3_L2_3e-06_run_2/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.0_momentum_0.3_L2_3e-06_run_2/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Test_eval.csv",
                       "Def_DETR_Plot_Analysis/03_Test_Dropout_Part_4/lr_0.001_warmup_None_dropout_0.0_momentum_0.3_L2_3e-06_run_3/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.0_momentum_0.3_L2_3e-06_run_3/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Test_eval.csv",
                       "Def_DETR_Plot_Analysis/03_Test_Dropout_Part_4/lr_0.001_warmup_None_dropout_0.0_momentum_0.3_L2_3e-06_run_4/evals_f1_score/Def_DETR_aug_True/lr_0.001_warmup_None_dropout_0.0_momentum_0.3_L2_3e-06_run_4/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Test_eval.csv"
                       ]
eval_csvs_config_detr_Test = list_eval_full_path

list_input_mode = ["Train", "Val", "Test"]
model_name = "Def DETR"
mode = "single"
gt_vers = ["regular", "regular", "maj"]
output = "Def_DETR_Plot_Analysis/Plots"

output_main_path = "Def_DETR_Plot_Analysis/03_Test_Dropout_Part_4/config_Train_"
_, _, output_csv_mean_config_detr_Train, output_csv_std_config_detr_Train = load_eval_csv_create_avg(
    eval_csvs=eval_csvs_config_detr_Train, output_main_path=output_main_path)

output_main_path = "Def_DETR_Plot_Analysis/03_Test_Dropout_Part_2/config_Val_"
_, _, output_csv_mean_config_detr_Val, output_csv_std_config_detr_Val = load_eval_csv_create_avg(
    eval_csvs=eval_csvs_config_detr_Val, output_main_path=output_main_path)

output_main_path = "Def_DETR_Plot_Analysis/03_Test_Dropout_Part_4/config_Test_"
_, _, output_csv_mean_config_detr_Test, output_csv_std_config_detr_Test = load_eval_csv_create_avg(
    eval_csvs=eval_csvs_config_detr_Test, output_main_path=output_main_path)

legend = ["Train", "Val", "Test"]
plot_f1_score_comparison_quick([output_csv_mean_config_detr_Train, output_csv_mean_config_detr_Val,
                                output_csv_mean_config_detr_Test],
                               list_input_mode, model_name, mode,
                               gt_vers, legend,
                               # list_std_full_path=None,
                               list_std_full_path=[output_csv_std_config_detr_Train, output_csv_std_config_detr_Val,
                                                   output_csv_std_config_detr_Test],
                               output=output,
                               smooth=True, smoothing_size=10, ylim=[0.6, 0.90],
                               plot_name_postfix="Def_DETR_bias_and_variance_AVG_smooth_10")
