""" This script creates the plots for the single hyperparameter configuration runs
    for different data augmentations of all three models.
"""
from custom_plotting import plot_f1_score_comparison_quick


# # # Cascade-RCNN (GaussianBlur)
# list_eval_full_path = ["Cascade_RCNN_Plot_Analysis/01_Test_DA_step_0/no_DA/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_False/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/Cascade-RCNN_aug_False_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
#                        "Cascade_RCNN_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussianBlur_0_3_s_0_p_10/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
#                        "Cascade_RCNN_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussianBlur_0_7_s_0_p_10/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
#                        "Cascade_RCNN_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussianBlur_0_13_s_0_p_10/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
#                        "Cascade_RCNN_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussianBlur_0_19_s_0_p_10/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
#                        "Cascade_RCNN_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussianBlur_0_35_s_0_p_10/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv"
#                        ]
# list_input_mode = ["Val"]
# model_name = "Cascade R-CNN"
# mode = "single"
# gt_vers = ["regular"]
# legend = ["no DA", "GB kernel 3", "GB kernel 7", "GB kernel 13", "GB kernel 19", "GB kernel 35"]
# output = "Cascade_RCNN_Plot_Analysis/Plots"
# plot_f1_score_comparison_quick(list_eval_full_path, list_input_mode, model_name, mode, gt_vers, legend, output=output,
#                                smooth=True, smoothing_size=9, ylim=[0.7275, 0.78], plot_name_postfix="GaussianBlur")

# # # Cascade-RCNN (GaussNoise)
list_eval_full_path = ["Cascade_RCNN_Plot_Analysis/01_Test_DA_step_0/no_DA/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_False/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/Cascade-RCNN_aug_False_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
                       "Cascade_RCNN_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussNoise_0_1000_p_10/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
                       "Cascade_RCNN_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussNoise_0_2000_p_10/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
                       "Cascade_RCNN_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussNoise_0_10000_p_10/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
                       "Cascade_RCNN_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussNoise_0_15000_p_10/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
                       "Cascade_RCNN_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussNoise_0_25000_p_10/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv"
                       ]
list_input_mode = ["Val"]
model_name = "Cascade R-CNN"
mode = "single"
gt_vers = ["regular"]
legend = ["no DA", "GN var 1000", "GN var 2000", "GN var 10000", "GN var 15000", "GN var 25000"]
output = "Cascade_RCNN_Plot_Analysis/Plots"
plot_f1_score_comparison_quick(list_eval_full_path, list_input_mode, model_name, mode, gt_vers, legend, output=output,
                               smooth=True, smoothing_size=9, ylim=[0.72, 0.77], plot_name_postfix="GaussNoise")

# # # Cascade-RCNN (RandomBrightnessContrast)
# list_eval_full_path = ["Cascade_RCNN_Plot_Analysis/01_Test_DA_step_0/no_DA/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_False/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/Cascade-RCNN_aug_False_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
#                        "Cascade_RCNN_Plot_Analysis/01_Test_DA_step_0/pixel_DA_rbc_02_p_10/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
#                        "Cascade_RCNN_Plot_Analysis/01_Test_DA_step_0/pixel_DA_rbc_05_p_10/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
#                        "Cascade_RCNN_Plot_Analysis/01_Test_DA_step_0/pixel_DA_rbc_08_p_10/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv"
#                        ]
# list_input_mode = ["Val"]
# model_name = "Cascade R-CNN"
# mode = "single"
# gt_vers = ["regular"]
# legend = ["no DA", "RBC lim 0.2 p 1.0", "RBC lim 0.5 p 1.0", "RBC lim 0.8 p 1.0"]
# output = "Cascade_RCNN_Plot_Analysis/Plots"
# plot_f1_score_comparison_quick(list_eval_full_path, list_input_mode, model_name, mode, gt_vers, legend, output=output,
#                                smooth=True, smoothing_size=5, ylim=[0.71, 0.76],
#                                plot_name_postfix="RandomBrightnessContrast")

# # # Cascade-RCNN (spatial DA)
# list_eval_full_path = ["Cascade_RCNN_Plot_Analysis/01_Test_DA_step_0/no_DA/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_False/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/Cascade-RCNN_aug_False_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
#                        "Cascade_RCNN_Plot_Analysis/01_Test_DA_step_0/spatial_DA/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv"
#                        ]
# list_input_mode = ["Val"]
# model_name = "Cascade R-CNN"
# mode = "single"
# gt_vers = ["regular"]
# legend = ["no DA", "spatial DA"]
# output = "Cascade_RCNN_Plot_Analysis/Plots"
# plot_f1_score_comparison_quick(list_eval_full_path, list_input_mode, model_name, mode, gt_vers, legend, output=output,
#                                smooth=True, smoothing_size=6, ylim=[0.72, 0.82], plot_name_postfix="spatial_DA")

# # # Cascade-RCNN (mixed DA)
# list_eval_full_path = ["Cascade_RCNN_Plot_Analysis/01_Test_DA_step_0/no_DA/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_False/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/Cascade-RCNN_aug_False_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
#                        "Cascade_RCNN_Plot_Analysis/01_Test_DA_step_0/spatial_DA/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
#                        "Cascade_RCNN_Plot_Analysis/01_Test_DA_step_0/mixed_DA_GB_0_13_s_0_p_10_GN_0_500_p_00_RBC_05_p_05/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv"
#                        ]
# list_input_mode = ["Val"]
# model_name = "Cascade R-CNN"
# mode = "single"
# gt_vers = ["regular"]
# legend = ["no DA", "spatial DA", "mixed DA"]
# output = "Cascade_RCNN_Plot_Analysis/Plots"
# plot_f1_score_comparison_quick(list_eval_full_path, list_input_mode, model_name, mode, gt_vers, legend, output=output,
#                                smooth=True, smoothing_size=3, ylim=[0.72, 0.82], plot_name_postfix="mixed_DA")


# # # Cascade-RCNN (Randomness Check)
# list_eval_full_path = ["Cascade_RCNN_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussianBlur_0_13_s_0_p_10/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
#                        "Cascade_RCNN_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussianBlur_Randomness_Check_1/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
#                        "Cascade_RCNN_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussianBlur_Randomness_Check_2/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
#                        "Cascade_RCNN_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussianBlur_Randomness_Check_3/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
#                        "Cascade_RCNN_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussianBlur_Randomness_Check_4/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv"
#                        ]
# list_input_mode = ["Val"]
# model_name = "Cascade R-CNN"
# mode = "single"
# gt_vers = ["regular"]
# legend = ["no set seed", "seed check 1", "seed check 2", "seed check 3", "seed check 4"]
# output = "Cascade_RCNN_Plot_Analysis/Plots"
# plot_f1_score_comparison_quick(list_eval_full_path, list_input_mode, model_name, mode, gt_vers, legend, output=output,
#                                smooth=True, smoothing_size=1, ylim=[0.72, 0.82], plot_name_postfix="Randomness_Check")
# 1: random python random seed
# 2: seed_data = seed_weight = [0], random.seed = imgaug.seed = [42]
# 3: seed_data = seed_weight = [0], random.seed = imgaug.seed = [42]
# 4: seed_data = seed_weight = [5], random.seed = imgaug.seed = [42]
# 5: seed_data = seed_weight = [5], random.seed = imgaug.seed = [11]


# # # Def-DETR (GaussianBlur)
# list_eval_full_path = ["Def_DETR_Plot_Analysis/01_Test_DA_step_0/no_DA/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Def_DETR_aug_False/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/Def_DETR_aug_False_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
#                        "Def_DETR_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussianBlur_0_3_s_0_p_10/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.3_Val_eval.csv",
#                        "Def_DETR_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussianBlur_0_7_s_0_p_10/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
#                        "Def_DETR_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussianBlur_0_13_s_0_p_10/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
#                        "Def_DETR_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussianBlur_0_19_s_0_p_10/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
#                        "Def_DETR_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussianBlur_0_35_s_0_p_10/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv"
#                        ]
# list_input_mode = ["Val"]
# model_name = "Def DETR"
# mode = "single"
# gt_vers = ["regular"]
# legend = ["no DA", "GB kernel 3", "GB kernel 7", "GB kernel 13", "GB kernel 19", "GB kernel 35"]
# output = "Def_DETR_Plot_Analysis/Plots"
# plot_f1_score_comparison_quick(list_eval_full_path, list_input_mode, model_name, mode, gt_vers, legend, output=output,
#                                smooth=True, smoothing_size=1, ylim=[0.0, 0.9], plot_name_postfix="GaussianBlur_no_smoothing_global")

# # # Def-DETR (GaussNoise)
list_eval_full_path = ["Def_DETR_Plot_Analysis/01_Test_DA_step_0/no_DA/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Def_DETR_aug_False/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/Def_DETR_aug_False_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       # "Def_DETR_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussNoise_0_100_p_10/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       # "Def_DETR_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussNoise_0_500_p_10/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       # "Def_DETR_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussNoise_0_1000_p_10/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussNoise_0_2000_p_10/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       # "Def_DETR_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussNoise_0_3000_p_10/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussNoise_0_10000_p_10/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussNoise_0_15000_p_10/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussNoise_0_25000_p_10/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv"
                       ]
list_input_mode = ["Val"]
model_name = "Def DETR"
mode = "single"
gt_vers = ["regular"]
legend = ["no DA", "GN var 2000", "GN var 10000", "GN var 15000", "GN var 25000"]
output = "Def_DETR_Plot_Analysis/Plots"
plot_f1_score_comparison_quick(list_eval_full_path, list_input_mode, model_name, mode, gt_vers, legend, output=output,
                               smooth=True, smoothing_size=5, ylim=[0.7, 0.76], plot_name_postfix="GaussNoise")


# # # Def-DETR (RandomBrightnessContrast)
# list_eval_full_path = ["Def_DETR_Plot_Analysis/01_Test_DA_step_0/no_DA/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Def_DETR_aug_False/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/Def_DETR_aug_False_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
#                        "Def_DETR_Plot_Analysis/01_Test_DA_step_0/pixel_DA_rbc_02_p_10/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
#                        "Def_DETR_Plot_Analysis/01_Test_DA_step_0/pixel_DA_rbc_05_p_10/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
#                        "Def_DETR_Plot_Analysis/01_Test_DA_step_0/pixel_DA_rbc_08_p_10/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv"
#                        ]
# list_input_mode = ["Val"]
# model_name = "Def DETR"
# mode = "single"
# gt_vers = ["regular"]
# legend = ["no DA", "RBC lim 0.2 p 1.0", "RBC lim 0.5 p 1.0", "RBC lim 0.8 p 1.0"]
# output = "Def_DETR_Plot_Analysis/Plots"
# plot_f1_score_comparison_quick(list_eval_full_path, list_input_mode, model_name, mode, gt_vers, legend, output=output,
#                                smooth=True, smoothing_size=4, ylim=[0.7, 0.79],
#                                plot_name_postfix="RandomBrightnessContrast")

# # # Def-DETR (spatial DA)
# list_eval_full_path = ["Def_DETR_Plot_Analysis/01_Test_DA_step_0/no_DA/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Def_DETR_aug_False/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/Def_DETR_aug_False_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
#                        "Def_DETR_Plot_Analysis/01_Test_DA_step_0/spatial_DA/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv"
#                        ]
# list_input_mode = ["Val"]
# model_name = "Def DETR"
# mode = "single"
# gt_vers = ["regular"]
# legend = ["no DA", "spatial DA"]
# output = "Def_DETR_Plot_Analysis/Plots"
# plot_f1_score_comparison_quick(list_eval_full_path, list_input_mode, model_name, mode, gt_vers, legend, output=output,
#                                smooth=True, smoothing_size=4, ylim=[0.7, 0.82],
#                                plot_name_postfix="spatial_DA")


# # # Def-DETR (mixed DA)
# list_eval_full_path = ["Def_DETR_Plot_Analysis/01_Test_DA_step_0/no_DA/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Def_DETR_aug_False/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/Def_DETR_aug_False_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
#                        "Def_DETR_Plot_Analysis/01_Test_DA_step_0/spatial_DA/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
#                        "Def_DETR_Plot_Analysis/01_Test_DA_step_0/mixed_DA_GB_0_19_s_0_p_02_GN_0_2000_p_02_RBC_08_p_10/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv"
#                        ]
# list_input_mode = ["Val"]
# model_name = "Def DETR"
# mode = "single"
# gt_vers = ["regular"]
# legend = ["no DA", "spatial DA", "mixed DA"]
# output = "Def_DETR_Plot_Analysis/Plots"
# plot_f1_score_comparison_quick(list_eval_full_path, list_input_mode, model_name, mode, gt_vers, legend, output=output,
#                                smooth=True, smoothing_size=4, ylim=[0.7, 0.83],
#                                plot_name_postfix="mixed_DA")



# # # VFNet (GaussianBlur)
# list_eval_full_path = ["VFNet_Plot_Analysis/01_Test_DA_step_0/no_DA/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/VFNet_aug_False/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/VFNet_aug_False_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
#                        "VFNet_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussianBlur_0_3_s_0_p_10/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
#                        "VFNet_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussianBlur_0_7_s_0_p_10/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
#                        "VFNet_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussianBlur_0_13_s_0_p_10/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
#                        "VFNet_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussianBlur_0_19_s_0_p_10/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv",
#                        "VFNet_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussianBlur_0_35_s_0_p_10/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.6_Val_eval.csv"
#                        ]
# list_input_mode = ["Val"]
# model_name = "VFNet"
# mode = "single"
# gt_vers = ["regular"]
# legend = ["no DA", "GB kernel 3", "GB kernel 7", "GB kernel 13", "GB kernel 19", "GB kernel 35"]
# output = "VFNet_Plot_Analysis/Plots"
# plot_f1_score_comparison_quick(list_eval_full_path, list_input_mode, model_name, mode, gt_vers, legend, output=output,
#                                smooth=True, smoothing_size=4, ylim=[0.6, 0.8], plot_name_postfix="GaussianBlur")

# # # VFNet (GaussNoise)
list_eval_full_path = ["VFNet_Plot_Analysis/01_Test_DA_step_0/no_DA/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/VFNet_aug_False/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/VFNet_aug_False_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       # "VFNet_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussNoise_0_100_p_10/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       # "VFNet_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussNoise_0_500_p_10/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "VFNet_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussNoise_0_2000_p_10/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "VFNet_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussNoise_0_10000_p_10/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "VFNet_Plot_Analysis/01_Test_DA_step_0/pixel_DA_gaussNoise_0_15000_p_10/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv"
                       ]
list_input_mode = ["Val"]
model_name = "VFNet"
mode = "single"
gt_vers = ["regular"]
legend = ["no DA", "GN var 2000", "GN var 10000", "GN var 15000"]
output = "VFNet_Plot_Analysis/Plots"
plot_f1_score_comparison_quick(list_eval_full_path, list_input_mode, model_name, mode, gt_vers, legend, output=output,
                               smooth=True, smoothing_size=4, ylim=[0.7, 0.8], plot_name_postfix="GaussNoise")

# # # VFNet (RandomBrightnessContrast)
# list_eval_full_path = ["VFNet_Plot_Analysis/01_Test_DA_step_0/no_DA/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/VFNet_aug_False/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/VFNet_aug_False_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
#                        "VFNet_Plot_Analysis/01_Test_DA_step_0/pixel_DA_rbc_02_p_10/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.4_Val_eval.csv",
#                        "VFNet_Plot_Analysis/01_Test_DA_step_0/pixel_DA_rbc_05_p_10/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
#                        "VFNet_Plot_Analysis/01_Test_DA_step_0/pixel_DA_rbc_08_p_10/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv"
#                        ]
# list_input_mode = ["Val"]
# model_name = "VFNet"
# mode = "single"
# gt_vers = ["regular"]
# legend = ["no DA", "RBC lim 0.2 p 1.0", "RBC lim 0.5 p 1.0", "RBC lim 0.8 p 1.0"]
# output = "VFNet_Plot_Analysis/Plots"
# plot_f1_score_comparison_quick(list_eval_full_path, list_input_mode, model_name, mode, gt_vers, legend, output=output,
#                                smooth=True, smoothing_size=4, ylim=[0.7, 0.8],
#                                plot_name_postfix="RandomBrightnessContrast")

# # # VFNet (spatial DA)
# list_eval_full_path = ["VFNet_Plot_Analysis/01_Test_DA_step_0/no_DA/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/VFNet_aug_False/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/VFNet_aug_False_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
#                        "VFNet_Plot_Analysis/01_Test_DA_step_0/spatial_DA/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv"
#                        ]
# list_input_mode = ["Val"]
# model_name = "VFNet"
# mode = "single"
# gt_vers = ["regular"]
# legend = ["no DA", "spatial DA"]
# output = "VFNet_Plot_Analysis/Plots"
# plot_f1_score_comparison_quick(list_eval_full_path, list_input_mode, model_name, mode, gt_vers, legend, output=output,
#                                smooth=True, smoothing_size=4, ylim=[0.7, 0.82],
#                                plot_name_postfix="spatial_DA")
