""" This script creates the plots for the multiple hyperparameter configuration runs
    for different sources of randomness of all three models, which create fluctuations in the learning curves.
"""
from custom_plotting import plot_f1_score_comparison_quick, load_eval_csv_create_avg


# # # Cascade-RCNN (all_seeds_fixed)
list_eval_full_path = ["Cascade_RCNN_Plot_Analysis/00_Test_Fluctuations/all_seeds_fixed/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.7_Val_eval.csv",
                       "Cascade_RCNN_Plot_Analysis/00_Test_Fluctuations/all_seeds_fixed/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_2/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_2/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.7_Val_eval.csv",
                       "Cascade_RCNN_Plot_Analysis/00_Test_Fluctuations/all_seeds_fixed/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_3/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_3/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.7_Val_eval.csv",
                       "Cascade_RCNN_Plot_Analysis/00_Test_Fluctuations/all_seeds_fixed/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_4/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_4/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.7_Val_eval.csv",
                       "Cascade_RCNN_Plot_Analysis/00_Test_Fluctuations/all_seeds_fixed/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_5/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_5/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.7_Val_eval.csv"
                       ]
list_input_mode = ["Val"]
model_name = "Cascade R-CNN"
mode = "single"
gt_vers = ["regular"]
legend = ["run 1", "run 2", "run 3", "run 4", "run 5"]
output = "Cascade_RCNN_Plot_Analysis/Plots"
plot_f1_score_comparison_quick(list_eval_full_path, list_input_mode, model_name, mode, gt_vers, legend, output=output,
                               smooth=True, smoothing_size=1, ylim=[0.0, 0.84], plot_name_postfix="all_seeds_fixed")

# # # Cascade-RCNN (all_seeds_fixed) (AVG)
eval_csvs = list_eval_full_path
output_main_path = "Cascade_RCNN_Plot_Analysis/00_Test_Fluctuations/all_seeds_fixed"
_, _, output_csv_mean, output_csv_std = load_eval_csv_create_avg(eval_csvs=eval_csvs, output_main_path=output_main_path)
legend = ["all seeds fixed"]
plot_f1_score_comparison_quick([output_csv_mean], list_input_mode, model_name, mode, gt_vers, legend,
                               list_std_full_path=[output_csv_std], output=output,
                               smooth=True, smoothing_size=1, ylim=[0.6, 0.84], plot_name_postfix="all_seeds_fixed_AVG")


# # # Cascade-RCNN (toggle_data_aug_seed)
list_eval_full_path = ["Cascade_RCNN_Plot_Analysis/00_Test_Fluctuations/toggle_data_aug_seed/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.7_Val_eval.csv",
                       "Cascade_RCNN_Plot_Analysis/00_Test_Fluctuations/toggle_data_aug_seed/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_2/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_2/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.7_Val_eval.csv",
                       "Cascade_RCNN_Plot_Analysis/00_Test_Fluctuations/toggle_data_aug_seed/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_3/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_3/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.7_Val_eval.csv",
                       "Cascade_RCNN_Plot_Analysis/00_Test_Fluctuations/toggle_data_aug_seed/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_4/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_4/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.7_Val_eval.csv",
                       "Cascade_RCNN_Plot_Analysis/00_Test_Fluctuations/toggle_data_aug_seed/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_5/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_5/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.7_Val_eval.csv"
                       ]
list_input_mode = ["Val"]
model_name = "Cascade R-CNN"
mode = "single"
gt_vers = ["regular"]
legend = ["run 1", "run 2", "run 3", "run 4", "run 5"]
output = "Cascade_RCNN_Plot_Analysis/Plots"
plot_f1_score_comparison_quick(list_eval_full_path, list_input_mode, model_name, mode, gt_vers, legend, output=output,
                               smooth=True, smoothing_size=1, ylim=[0.0, 0.84], plot_name_postfix="toggle_data_aug_seed")

# # # Cascade-RCNN (toggle_data_aug_seed) (AVG)
eval_csvs = list_eval_full_path
output_main_path = "Cascade_RCNN_Plot_Analysis/00_Test_Fluctuations/toggle_data_aug_seed"
_, _, output_csv_mean, output_csv_std = load_eval_csv_create_avg(eval_csvs=eval_csvs, output_main_path=output_main_path)
legend = ["tweak data augmentation seed"]
plot_f1_score_comparison_quick([output_csv_mean], list_input_mode, model_name, mode, gt_vers, legend,
                               list_std_full_path=[output_csv_std], output=output,
                               smooth=True, smoothing_size=1, ylim=[0.6, 0.84],
                               plot_name_postfix="toggle_data_aug_seed_AVG")


# # # Cascade-RCNN (toggle_data_seed)
list_eval_full_path = ["Cascade_RCNN_Plot_Analysis/00_Test_Fluctuations/toggle_data_seed/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.7_Val_eval.csv",
                       "Cascade_RCNN_Plot_Analysis/00_Test_Fluctuations/toggle_data_seed/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_2/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_2/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.7_Val_eval.csv",
                       "Cascade_RCNN_Plot_Analysis/00_Test_Fluctuations/toggle_data_seed/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_3/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_3/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.7_Val_eval.csv",
                       "Cascade_RCNN_Plot_Analysis/00_Test_Fluctuations/toggle_data_seed/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_4/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_4/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.7_Val_eval.csv",
                       "Cascade_RCNN_Plot_Analysis/00_Test_Fluctuations/toggle_data_seed/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_5/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_5/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.7_Val_eval.csv"
                       ]
list_input_mode = ["Val"]
model_name = "Cascade R-CNN"
mode = "single"
gt_vers = ["regular"]
legend = ["run 1", "run 2", "run 3", "run 4", "run 5"]
output = "Cascade_RCNN_Plot_Analysis/Plots"
plot_f1_score_comparison_quick(list_eval_full_path, list_input_mode, model_name, mode, gt_vers, legend, output=output,
                               smooth=True, smoothing_size=1, ylim=[0.0, 0.84], plot_name_postfix="toggle_data_seed")

# # # Cascade-RCNN (toggle_data_seed) (AVG)
eval_csvs = list_eval_full_path
output_main_path = "Cascade_RCNN_Plot_Analysis/00_Test_Fluctuations/toggle_data_seed"
_, _, output_csv_mean, output_csv_std = load_eval_csv_create_avg(eval_csvs=eval_csvs, output_main_path=output_main_path)
legend = ["tweak data sampling seed"]
plot_f1_score_comparison_quick([output_csv_mean], list_input_mode, model_name, mode, gt_vers, legend,
                               list_std_full_path=[output_csv_std], output=output,
                               smooth=True, smoothing_size=1, ylim=[0.6, 0.84],
                               plot_name_postfix="toggle_data_seed_AVG")


# # # Cascade-RCNN (toggle_weight_seed)
list_eval_full_path = ["Cascade_RCNN_Plot_Analysis/00_Test_Fluctuations/toggle_weight_seed/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.7_Val_eval.csv",
                       "Cascade_RCNN_Plot_Analysis/00_Test_Fluctuations/toggle_weight_seed/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_2/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_2/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.7_Val_eval.csv",
                       "Cascade_RCNN_Plot_Analysis/00_Test_Fluctuations/toggle_weight_seed/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_3/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_3/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.7_Val_eval.csv",
                       "Cascade_RCNN_Plot_Analysis/00_Test_Fluctuations/toggle_weight_seed/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_4/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_4/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.7_Val_eval.csv",
                       "Cascade_RCNN_Plot_Analysis/00_Test_Fluctuations/toggle_weight_seed/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_5/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_5/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.7_Val_eval.csv"
                       ]
list_input_mode = ["Val"]
model_name = "Cascade R-CNN"
mode = "single"
gt_vers = ["regular"]
legend = ["run 1", "run 2", "run 3", "run 4", "run 5"]
output = "Cascade_RCNN_Plot_Analysis/Plots"
plot_f1_score_comparison_quick(list_eval_full_path, list_input_mode, model_name, mode, gt_vers, legend, output=output,
                               smooth=True, smoothing_size=1, ylim=[0.0, 0.84], plot_name_postfix="toggle_weight_seed")

# # # Cascade-RCNN (toggle_weight_seed) (AVG)
eval_csvs = list_eval_full_path
output_main_path = "Cascade_RCNN_Plot_Analysis/00_Test_Fluctuations/toggle_weight_seed"
_, _, output_csv_mean, output_csv_std = load_eval_csv_create_avg(eval_csvs=eval_csvs, output_main_path=output_main_path)
legend = ["tweak initial weights seed"]
plot_f1_score_comparison_quick([output_csv_mean], list_input_mode, model_name, mode, gt_vers, legend,
                               list_std_full_path=[output_csv_std], output=output,
                               smooth=True, smoothing_size=1, ylim=[0.6, 0.84],
                               plot_name_postfix="toggle_weight_seed_AVG")


# # # Cascade-RCNN (all_seeds_random)
list_eval_full_path = ["Cascade_RCNN_Plot_Analysis/00_Test_Fluctuations/all_seeds_random/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_1/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.7_Val_eval.csv",
                       "Cascade_RCNN_Plot_Analysis/00_Test_Fluctuations/all_seeds_random/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_2/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_2/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.7_Val_eval.csv",
                       "Cascade_RCNN_Plot_Analysis/00_Test_Fluctuations/all_seeds_random/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_3/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_3/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.7_Val_eval.csv",
                       "Cascade_RCNN_Plot_Analysis/00_Test_Fluctuations/all_seeds_random/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_4/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_4/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.7_Val_eval.csv",
                       "Cascade_RCNN_Plot_Analysis/00_Test_Fluctuations/all_seeds_random/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_5/evals_f1_score/Cascade-RCNN_aug_True/lr_0.0001_warmup_None_momentum_0.9_L2_0.0003_run_5/Cascade-RCNN_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.7_Val_eval.csv"
                       ]
list_input_mode = ["Val"]
model_name = "Cascade R-CNN"
mode = "single"
gt_vers = ["regular"]
legend = ["run 1", "run 2", "run 3", "run 4", "run 5"]
output = "Cascade_RCNN_Plot_Analysis/Plots"
plot_f1_score_comparison_quick(list_eval_full_path, list_input_mode, model_name, mode, gt_vers, legend, output=output,
                               smooth=True, smoothing_size=1, ylim=[0.0, 0.84], plot_name_postfix="all_seeds_random")

# # # Cascade-RCNN (all_seeds_random) (AVG)
eval_csvs = list_eval_full_path
output_main_path = "Cascade_RCNN_Plot_Analysis/00_Test_Fluctuations/all_seeds_random"
_, _, output_csv_mean, output_csv_std = load_eval_csv_create_avg(eval_csvs=eval_csvs, output_main_path=output_main_path)
legend = ["all seeds random"]
plot_f1_score_comparison_quick([output_csv_mean], list_input_mode, model_name, mode, gt_vers, legend,
                               list_std_full_path=[output_csv_std], output=output,
                               smooth=True, smoothing_size=1, ylim=[0.6, 0.84],
                               plot_name_postfix="all_seeds_random_AVG")


# # # Def-DETR (all_seeds_fixed)
list_eval_full_path = ["Def_DETR_Plot_Analysis/00_Test_Fluctuations/all_seeds_fixed/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/00_Test_Fluctuations/all_seeds_fixed/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_2/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_2/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/00_Test_Fluctuations/all_seeds_fixed/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_3/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_3/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/00_Test_Fluctuations/all_seeds_fixed/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_4/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_4/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/00_Test_Fluctuations/all_seeds_fixed/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_5/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_5/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv"
                       ]
list_input_mode = ["Val"]
model_name = "Def DETR"
mode = "single"
gt_vers = ["regular"]
legend = ["run 1", "run 2", "run 3", "run 4", "run 5"]
output = "Def_DETR_Plot_Analysis/Plots"
plot_f1_score_comparison_quick(list_eval_full_path, list_input_mode, model_name, mode, gt_vers, legend, output=output,
                               smooth=True, smoothing_size=1, ylim=[0.0, 0.84], plot_name_postfix="all_seeds_fixed")

# # # Def-DETR (all_seeds_fixed) (AVG)
eval_csvs = list_eval_full_path
output_main_path = "Def_DETR_Plot_Analysis/00_Test_Fluctuations/all_seeds_fixed"
_, _, output_csv_mean, output_csv_std = load_eval_csv_create_avg(eval_csvs=eval_csvs, output_main_path=output_main_path)
legend = ["all seeds fixed"]
plot_f1_score_comparison_quick([output_csv_mean], list_input_mode, model_name, mode, gt_vers, legend,
                               list_std_full_path=[output_csv_std], output=output,
                               smooth=True, smoothing_size=1, ylim=[0.6, 0.84], plot_name_postfix="all_seeds_fixed_AVG")


# # # Def-DETR (toggle_data_aug_seed)
list_eval_full_path = ["Def_DETR_Plot_Analysis/00_Test_Fluctuations/toggle_data_aug_seed/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/00_Test_Fluctuations/toggle_data_aug_seed/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_2/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_2/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/00_Test_Fluctuations/toggle_data_aug_seed/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_3/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_3/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/00_Test_Fluctuations/toggle_data_aug_seed/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_4/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_4/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/00_Test_Fluctuations/toggle_data_aug_seed/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_5/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_5/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv"
                       ]
list_input_mode = ["Val"]
model_name = "Def DETR"
mode = "single"
gt_vers = ["regular"]
legend = ["run 1", "run 2", "run 3", "run 4", "run 5"]
output = "Def_DETR_Plot_Analysis/Plots"
plot_f1_score_comparison_quick(list_eval_full_path, list_input_mode, model_name, mode, gt_vers, legend, output=output,
                               smooth=True, smoothing_size=1, ylim=[0.0, 0.84], plot_name_postfix="toggle_data_aug_seed")

# # # Def-DETR (toggle_data_aug_seed) (AVG)
eval_csvs = list_eval_full_path
output_main_path = "Def_DETR_Plot_Analysis/00_Test_Fluctuations/toggle_data_aug_seed"
_, _, output_csv_mean, output_csv_std = load_eval_csv_create_avg(eval_csvs=eval_csvs, output_main_path=output_main_path)
legend = ["tweak data augmentation seed"]
plot_f1_score_comparison_quick([output_csv_mean], list_input_mode, model_name, mode, gt_vers, legend,
                               list_std_full_path=[output_csv_std], output=output,
                               smooth=True, smoothing_size=1, ylim=[0.6, 0.84],
                               plot_name_postfix="toggle_data_aug_seed_AVG")


# # # Def-DETR (toggle_data_seed)
list_eval_full_path = ["Def_DETR_Plot_Analysis/00_Test_Fluctuations/toggle_data_seed/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/00_Test_Fluctuations/toggle_data_seed/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_2/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_2/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/00_Test_Fluctuations/toggle_data_seed/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_3/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_3/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/00_Test_Fluctuations/toggle_data_seed/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_4/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_4/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/00_Test_Fluctuations/toggle_data_seed/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_5/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_5/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv"
                       ]
list_input_mode = ["Val"]
model_name = "Def DETR"
mode = "single"
gt_vers = ["regular"]
legend = ["run 1", "run 2", "run 3", "run 4", "run 5"]
output = "Def_DETR_Plot_Analysis/Plots"
plot_f1_score_comparison_quick(list_eval_full_path, list_input_mode, model_name, mode, gt_vers, legend, output=output,
                               smooth=True, smoothing_size=1, ylim=[0.0, 0.84], plot_name_postfix="toggle_data_seed")

# # # Def-DETR (toggle_data_seed) (AVG)
eval_csvs = list_eval_full_path
output_main_path = "Def_DETR_Plot_Analysis/00_Test_Fluctuations/toggle_data_seed"
_, _, output_csv_mean, output_csv_std = load_eval_csv_create_avg(eval_csvs=eval_csvs, output_main_path=output_main_path)
legend = ["tweak data sampling seed"]
plot_f1_score_comparison_quick([output_csv_mean], list_input_mode, model_name, mode, gt_vers, legend,
                               list_std_full_path=[output_csv_std], output=output,
                               smooth=True, smoothing_size=1, ylim=[0.6, 0.84], plot_name_postfix="toggle_data_seed_AVG")


# # # Def-DETR (toggle_weight_seed)
list_eval_full_path = ["Def_DETR_Plot_Analysis/00_Test_Fluctuations/toggle_weight_seed/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/00_Test_Fluctuations/toggle_weight_seed/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_2/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_2/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/00_Test_Fluctuations/toggle_weight_seed/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_3/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_3/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/00_Test_Fluctuations/toggle_weight_seed/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_4/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_4/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/00_Test_Fluctuations/toggle_weight_seed/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_5/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_5/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv"
                       ]
list_input_mode = ["Val"]
model_name = "Def DETR"
mode = "single"
gt_vers = ["regular"]
legend = ["run 1", "run 2", "run 3", "run 4", "run 5"]
output = "Def_DETR_Plot_Analysis/Plots"
plot_f1_score_comparison_quick(list_eval_full_path, list_input_mode, model_name, mode, gt_vers, legend, output=output,
                               smooth=True, smoothing_size=1, ylim=[0.0, 0.84], plot_name_postfix="toggle_weight_seed")

# # # Def-DETR (toggle_weight_seed) (AVG)
eval_csvs = list_eval_full_path
output_main_path = "Def_DETR_Plot_Analysis/00_Test_Fluctuations/toggle_weight_seed"
_, _, output_csv_mean, output_csv_std = load_eval_csv_create_avg(eval_csvs=eval_csvs, output_main_path=output_main_path)
legend = ["tweak initial weights seed"]
plot_f1_score_comparison_quick([output_csv_mean], list_input_mode, model_name, mode, gt_vers, legend,
                               list_std_full_path=[output_csv_std], output=output,
                               smooth=True, smoothing_size=1, ylim=[0.6, 0.84],
                               plot_name_postfix="toggle_weight_seed_AVG")


# # # Def-DETR (all_seeds_random)
list_eval_full_path = ["Def_DETR_Plot_Analysis/00_Test_Fluctuations/all_seeds_random/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_1/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/00_Test_Fluctuations/all_seeds_random/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_2/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_2/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/00_Test_Fluctuations/all_seeds_random/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_3/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_3/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/00_Test_Fluctuations/all_seeds_random/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_4/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_4/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "Def_DETR_Plot_Analysis/00_Test_Fluctuations/all_seeds_random/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_5/evals_f1_score/Def_DETR_aug_True/lr_1e-05_warmup_None_dropout_0.1_momentum_0.9_L2_0.0003_run_5/Def_DETR_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv"
                       ]
list_input_mode = ["Val"]
model_name = "Def DETR"
mode = "single"
gt_vers = ["regular"]
legend = ["run 1", "run 2", "run 3", "run 4", "run 5"]
output = "Def_DETR_Plot_Analysis/Plots"
plot_f1_score_comparison_quick(list_eval_full_path, list_input_mode, model_name, mode, gt_vers, legend, output=output,
                               smooth=True, smoothing_size=1, ylim=[0.0, 0.84], plot_name_postfix="all_seeds_random")

# # # Def-DETR (all_seeds_random) (AVG)
eval_csvs = list_eval_full_path
output_main_path = "Def_DETR_Plot_Analysis/00_Test_Fluctuations/all_seeds_random"
_, _, output_csv_mean, output_csv_std = load_eval_csv_create_avg(eval_csvs=eval_csvs, output_main_path=output_main_path)
legend = ["all seeds random"]
plot_f1_score_comparison_quick([output_csv_mean], list_input_mode, model_name, mode, gt_vers, legend,
                               list_std_full_path=[output_csv_std], output=output,
                               smooth=True, smoothing_size=1, ylim=[0.6, 0.84],
                               plot_name_postfix="all_seeds_random_AVG")


# # # VFNet (all_seeds_fixed)
list_eval_full_path = ["VFNet_Plot_Analysis/00_Test_Fluctuations/all_seeds_fixed/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "VFNet_Plot_Analysis/00_Test_Fluctuations/all_seeds_fixed/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_2/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_2/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "VFNet_Plot_Analysis/00_Test_Fluctuations/all_seeds_fixed/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_3/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_3/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "VFNet_Plot_Analysis/00_Test_Fluctuations/all_seeds_fixed/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_4/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_4/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "VFNet_Plot_Analysis/00_Test_Fluctuations/all_seeds_fixed/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_5/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_5/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv"
                       ]
list_input_mode = ["Val"]
model_name = "VFNet"
mode = "single"
gt_vers = ["regular"]
legend = ["run 1", "run 2", "run 3", "run 4", "run 5"]
output = "VFNet_Plot_Analysis/Plots"
plot_f1_score_comparison_quick(list_eval_full_path, list_input_mode, model_name, mode, gt_vers, legend, output=output,
                               smooth=True, smoothing_size=1, ylim=[0.0, 0.84], plot_name_postfix="all_seeds_fixed")

# # # VFNet (all_seeds_fixed) (AVG)
eval_csvs = list_eval_full_path
output_main_path = "VFNet_Plot_Analysis/00_Test_Fluctuations/all_seeds_fixed"
_, _, output_csv_mean, output_csv_std = load_eval_csv_create_avg(eval_csvs=eval_csvs, output_main_path=output_main_path)
legend = ["all seeds fixed"]
plot_f1_score_comparison_quick([output_csv_mean], list_input_mode, model_name, mode, gt_vers, legend,
                               list_std_full_path=[output_csv_std], output=output,
                               smooth=True, smoothing_size=1, ylim=[0.6, 0.84], plot_name_postfix="all_seeds_fixed_AVG")


# # # VFNet (toggle_data_aug_seed)
list_eval_full_path = ["VFNet_Plot_Analysis/00_Test_Fluctuations/toggle_data_aug_seed/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "VFNet_Plot_Analysis/00_Test_Fluctuations/toggle_data_aug_seed/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_2/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_2/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "VFNet_Plot_Analysis/00_Test_Fluctuations/toggle_data_aug_seed/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_3/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_3/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "VFNet_Plot_Analysis/00_Test_Fluctuations/toggle_data_aug_seed/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_4/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_4/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "VFNet_Plot_Analysis/00_Test_Fluctuations/toggle_data_aug_seed/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_5/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_5/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv"
                       ]
list_input_mode = ["Val"]
model_name = "VFNet"
mode = "single"
gt_vers = ["regular"]
legend = ["run 1", "run 2", "run 3", "run 4", "run 5"]
output = "VFNet_Plot_Analysis/Plots"
plot_f1_score_comparison_quick(list_eval_full_path, list_input_mode, model_name, mode, gt_vers, legend, output=output,
                               smooth=True, smoothing_size=1, ylim=[0.0, 0.84], plot_name_postfix="toggle_data_aug_seed")

# # # VFNet (toggle_data_aug_seed) (AVG)
eval_csvs = list_eval_full_path
output_main_path = "VFNet_Plot_Analysis/00_Test_Fluctuations/toggle_data_aug_seed"
_, _, output_csv_mean, output_csv_std = load_eval_csv_create_avg(eval_csvs=eval_csvs, output_main_path=output_main_path)
legend = ["tweak data augmentation seed"]
plot_f1_score_comparison_quick([output_csv_mean], list_input_mode, model_name, mode, gt_vers, legend,
                               list_std_full_path=[output_csv_std], output=output,
                               smooth=True, smoothing_size=1, ylim=[0.6, 0.84],
                               plot_name_postfix="toggle_data_aug_seed_AVG")


# # # VFNet (toggle_data_seed)
list_eval_full_path = ["VFNet_Plot_Analysis/00_Test_Fluctuations/toggle_data_seed/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "VFNet_Plot_Analysis/00_Test_Fluctuations/toggle_data_seed/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_2/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_2/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "VFNet_Plot_Analysis/00_Test_Fluctuations/toggle_data_seed/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_3/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_3/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "VFNet_Plot_Analysis/00_Test_Fluctuations/toggle_data_seed/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_4/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_4/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "VFNet_Plot_Analysis/00_Test_Fluctuations/toggle_data_seed/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_5/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_5/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv"
                       ]
list_input_mode = ["Val"]
model_name = "VFNet"
mode = "single"
gt_vers = ["regular"]
legend = ["run 1", "run 2", "run 3", "run 4", "run 5"]
output = "VFNet_Plot_Analysis/Plots"
plot_f1_score_comparison_quick(list_eval_full_path, list_input_mode, model_name, mode, gt_vers, legend, output=output,
                               smooth=True, smoothing_size=1, ylim=[0.0, 0.84], plot_name_postfix="toggle_data_seed")

# # # VFNet (toggle_data_seed) (AVG)
eval_csvs = list_eval_full_path
output_main_path = "VFNet_Plot_Analysis/00_Test_Fluctuations/toggle_data_seed"
_, _, output_csv_mean, output_csv_std = load_eval_csv_create_avg(eval_csvs=eval_csvs, output_main_path=output_main_path)
legend = ["tweak data sampling seed"]
plot_f1_score_comparison_quick([output_csv_mean], list_input_mode, model_name, mode, gt_vers, legend,
                               list_std_full_path=[output_csv_std], output=output,
                               smooth=True, smoothing_size=1, ylim=[0.6, 0.84],
                               plot_name_postfix="toggle_data_seed_AVG")


# # # VFNet (toggle_weight_seed)
list_eval_full_path = ["VFNet_Plot_Analysis/00_Test_Fluctuations/toggle_weight_seed/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "VFNet_Plot_Analysis/00_Test_Fluctuations/toggle_weight_seed/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_2/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_2/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "VFNet_Plot_Analysis/00_Test_Fluctuations/toggle_weight_seed/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_3/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_3/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "VFNet_Plot_Analysis/00_Test_Fluctuations/toggle_weight_seed/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_4/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_4/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "VFNet_Plot_Analysis/00_Test_Fluctuations/toggle_weight_seed/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_5/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_5/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv"
                       ]
list_input_mode = ["Val"]
model_name = "VFNet"
mode = "single"
gt_vers = ["regular"]
legend = ["run 1", "run 2", "run 3", "run 4", "run 5"]
output = "VFNet_Plot_Analysis/Plots"
plot_f1_score_comparison_quick(list_eval_full_path, list_input_mode, model_name, mode, gt_vers, legend, output=output,
                               smooth=True, smoothing_size=1, ylim=[0.0, 0.84], plot_name_postfix="toggle_weight_seed")

# # # VFNet (toggle_weight_seed) (AVG)
eval_csvs = list_eval_full_path
output_main_path = "VFNet_Plot_Analysis/00_Test_Fluctuations/toggle_weight_seed"
_, _, output_csv_mean, output_csv_std = load_eval_csv_create_avg(eval_csvs=eval_csvs, output_main_path=output_main_path)
legend = ["tweak initial weights seed"]
plot_f1_score_comparison_quick([output_csv_mean], list_input_mode, model_name, mode, gt_vers, legend,
                               list_std_full_path=[output_csv_std], output=output,
                               smooth=True, smoothing_size=1, ylim=[0.6, 0.84],
                               plot_name_postfix="toggle_weight_seed_AVG")


# # # VFNet (all_seeds_random)
list_eval_full_path = ["VFNet_Plot_Analysis/00_Test_Fluctuations/all_seeds_random/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_1/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "VFNet_Plot_Analysis/00_Test_Fluctuations/all_seeds_random/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_2/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_2/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "VFNet_Plot_Analysis/00_Test_Fluctuations/all_seeds_random/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_3/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_3/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "VFNet_Plot_Analysis/00_Test_Fluctuations/all_seeds_random/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_4/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_4/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv",
                       "VFNet_Plot_Analysis/00_Test_Fluctuations/all_seeds_random/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_5/evals_f1_score/VFNet_aug_True/lr_1e-05_warmup_None_momentum_0.9_L2_0.0003_run_5/VFNet_aug_True_theta_0.5_delta_track_0.3_delta_eval_0.5_Val_eval.csv"
                       ]
list_input_mode = ["Val"]
model_name = "VFNet"
mode = "single"
gt_vers = ["regular"]
legend = ["run 1", "run 2", "run 3", "run 4", "run 5"]
output = "VFNet_Plot_Analysis/Plots"
plot_f1_score_comparison_quick(list_eval_full_path, list_input_mode, model_name, mode, gt_vers, legend, output=output,
                               smooth=True, smoothing_size=1, ylim=[0.0, 0.84], plot_name_postfix="all_seeds_random")

# # # VFNet (all_seeds_random) (AVG)
eval_csvs = list_eval_full_path
output_main_path = "VFNet_Plot_Analysis/00_Test_Fluctuations/all_seeds_random"
_, _, output_csv_mean, output_csv_std = load_eval_csv_create_avg(eval_csvs=eval_csvs, output_main_path=output_main_path)
legend = ["all seeds random"]
plot_f1_score_comparison_quick([output_csv_mean], list_input_mode, model_name, mode, gt_vers, legend,
                               list_std_full_path=[output_csv_std], output=output,
                               smooth=True, smoothing_size=1, ylim=[0.6, 0.84],
                               plot_name_postfix="all_seeds_random_AVG")
