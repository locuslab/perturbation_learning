##  MNIST experiments 
# Train L-infinity and RTS perturbation sets (Table 2 of paper)
python train.py --config configs/mnist_fc_linf.json 
python train.py --config configs/mnist_conv_linf.json 
python train.py --config configs/mnist_stn_rts.json
python train.py --config configs/mnist_stn_limited_1.json
python train.py --config configs/mnist_stn_limited_5.json

# Evaluate L-infinity and RTS perturbation sets (Table 2 of paper)
python eval.py --config configs/mnist_fc_linf.json --config-eval configs_eval/mnist_linf_fc_28.json
python eval.py --config configs/mnist_conv_linf.json --config-eval configs_eval/mnist_linf_conv_29.json 
python eval.py --config configs/mnist_stn_rts.json --config-eval configs_eval/mnist_rts_14.json
python eval.py --config configs/mnist_stn_limited_1.json --config-eval configs_eval/mnist_rts_14.json
python eval.py --config configs/mnist_stn_limited_5.json --config-eval configs_eval/mnist_rts_14.json

## CIFAR10 experiments
# Train common corruptions perturbation sets (Tables 5+6 of paper)
python train.py --config configs/cifar10_c_rectangle.json
python train.py --config configs/cifar10_c_rectangle_withoriginal.json
python train.py --config configs/cifar10_c_rectangle_nooriginal.json

# Evaluate common corruptions perturbation sets (Tables 5+6 of paper)
python eval.py --config configs/cifar10_c_rectangle.json --config-eval configs_eval/cifar10c_28.json
python eval.py --config configs/cifar10_c_rectangle_withoriginal.json --config-eval configs_eval/cifar10c_34.json
python eval.py --config configs/cifar10_c_rectangle_nooriginal.json --config-eval configs_eval/cifar10c_28.json

# Train classifier robust to common corruptions (Tables 1 of paper)
python robust_train.py --config-robust configs_robust/cifar10_wideresnet_sample_aug.json 
python robust_train.py --config-robust configs_robust/cifar10_wideresnet_cvae_aug.json 
python robust_train.py --config-robust configs_robust/cifar10_wideresnet_cvae_attack.json 

# Evaluate robustness of classifiers to common corruptions (Table 1 of paper)
python robust_eval.py --config-robust configs_robust/cifar10_wideresnet_sample_aug.json --config-attack configs_attack/cifar10_clean.json
python robust_eval.py --config-robust configs_robust/cifar10_wideresnet_sample_aug.json --config-attack configs_attack/cifar10c_clean.json
python robust_eval.py --config-robust configs_robust/cifar10_wideresnet_sample_aug.json --config-attack configs_attack/cifar10c_ood_clean.json
python robust_eval.py --config-robust configs_robust/cifar10_wideresnet_sample_aug.json --config-attack configs_attack/cifar10c_cvae_pgd_27.json
python robust_eval.py --config-robust configs_robust/cifar10_wideresnet_sample_aug.json --config-attack configs_attack/cifar10c_cvae_pgd_39.json
python robust_eval.py --config-robust configs_robust/cifar10_wideresnet_sample_aug.json --config-attack configs_attack/cifar10c_cvae_pgd_102.json

python robust_eval.py --config-robust configs_robust/cifar10_wideresnet_cvae_aug.json --config-attack configs_attack/cifar10c_clean.json
python robust_eval.py --config-robust configs_robust/cifar10_wideresnet_cvae_aug.json --config-attack configs_attack/cifar10c_ood_clean.json
python robust_eval.py --config-robust configs_robust/cifar10_wideresnet_cvae_aug.json --config-attack configs_attack/cifar10c_cvae_pgd_27.json
python robust_eval.py --config-robust configs_robust/cifar10_wideresnet_cvae_aug.json --config-attack configs_attack/cifar10c_cvae_pgd_39.json
python robust_eval.py --config-robust configs_robust/cifar10_wideresnet_cvae_aug.json --config-attack configs_attack/cifar10c_cvae_pgd_102.json
python robust_eval.py --config-robust configs_robust/cifar10_wideresnet_cvae_aug.json --config-attack configs_attack/cifar10_clean.json

python robust_eval.py --config-robust configs_robust/cifar10_wideresnet_cvae_attack.json --config-attack configs_attack/cifar10_clean.json
python robust_eval.py --config-robust configs_robust/cifar10_wideresnet_cvae_attack.json --config-attack configs_attack/cifar10c_clean.json
python robust_eval.py --config-robust configs_robust/cifar10_wideresnet_cvae_attack.json --config-attack configs_attack/cifar10c_ood_clean.json
python robust_eval.py --config-robust configs_robust/cifar10_wideresnet_cvae_attack.json --config-attack configs_attack/cifar10c_cvae_pgd_27.json
python robust_eval.py --config-robust configs_robust/cifar10_wideresnet_cvae_attack.json --config-attack configs_attack/cifar10c_cvae_pgd_39.json
python robust_eval.py --config-robust configs_robust/cifar10_wideresnet_cvae_attack.json --config-attack configs_attack/cifar10c_cvae_pgd_102.json

# Train certifiably robust classifiers with randomized smoothing (Table 7 of paper)
python robust_train.py --config-robust configs_robust/cifar10_wideresnet_cvae_smoothing_084.json 
python robust_train.py --config-robust configs_robust/cifar10_wideresnet_cvae_smoothing_122.json 
python robust_train.py --config-robust configs_robust/cifar10_wideresnet_cvae_smoothing_319.json 

# Evaluate performance and get certified radius of smoothed classifiers (Table 7 of paper)
python robust_train.py --config-robust configs_robust/cifar10_wideresnet_cvae_smoothing_084.json --config-attack configs_attack/cifar10c_smooth_perturbed_084.json
python robust_train.py --config-robust configs_robust/cifar10_wideresnet_cvae_smoothing_084.json --config-attack configs_attack/cifar10c_smooth_ood_084.json
python robust_train.py --config-robust configs_robust/cifar10_wideresnet_cvae_smoothing_084.json --config-attack configs_attack/cifar10c_smooth_certify_084.json

python robust_train.py --config-robust configs_robust/cifar10_wideresnet_cvae_smoothing_122.json --config-attack configs_attack/cifar10c_smooth_perturbed_122.json
python robust_train.py --config-robust configs_robust/cifar10_wideresnet_cvae_smoothing_122.json --config-attack configs_attack/cifar10c_smooth_ood_122.json
python robust_train.py --config-robust configs_robust/cifar10_wideresnet_cvae_smoothing_122.json --config-attack configs_attack/cifar10c_smooth_certify_122.json

python robust_train.py --config-robust configs_robust/cifar10_wideresnet_cvae_smoothing_319.json --config-attack configs_attack/cifar10c_smooth_perturbed_319.json
python robust_train.py --config-robust configs_robust/cifar10_wideresnet_cvae_smoothing_319.json --config-attack configs_attack/cifar10c_smooth_ood_319.json
python robust_train.py --config-robust configs_robust/cifar10_wideresnet_cvae_smoothing_319.json --config-attack configs_attack/cifar10c_smooth_certify_319.json

## Multi Illumination
# Train illumination perturbation sets at multiple resolutions (Table 8+9 of paper)
python train.py --config configs/mi_unet_mip5.json
python train.py --config configs/mi_unet_mip4.json
python train.py --config configs/mi_unet_mip3.json

# Evaluate illumination perturbation sets (Table 8+9 of paper)
python eval.py --config configs/mi_unet_mip5.json --config-eval configs_eval/mi_mip5_17.json
python eval.py --config configs/mi_unet_mip4.json --config-eval configs_eval/mi_mip4_25.json
python eval.py --config configs/mi_unet_mip3.json --config-eval configs_eval/mi_mip3_21.json

# Train segementation model robust to lighting perturbations (Table 10 of paper)
python robust_train.py --config-robust configs_robust/mi_unet_sample_first.json
python robust_train.py --config-robust configs_robust/mi_unet_sample_aug.json
python robust_train.py --config-robust configs_robust/mi_unet_cvae_aug.json
python robust_train.py --config-robust configs_robust/mi_unet_cvae_attack.json

# Evaluate robustness of segmentation model to lighting perturbations (Table 10 of paper)
python robust_train.py --config-robust configs_robust/mi_unet_sample_first.json --config-attack configs_attack/mi_clean.json
python robust_train.py --config-robust configs_robust/mi_unet_sample_first.json --config-attack configs_attack/mi_cvae_pgd_735.json
python robust_train.py --config-robust configs_robust/mi_unet_sample_first.json --config-attack configs_attack/mi_cvae_pgd_881.json
python robust_train.py --config-robust configs_robust/mi_unet_sample_first.json --config-attack configs_attack/mi_cvae_pgd.json

python robust_train.py --config-robust configs_robust/mi_unet_sample_aug.json --config-attack configs_attack/mi_clean.json
python robust_train.py --config-robust configs_robust/mi_unet_sample_aug.json --config-attack configs_attack/mi_cvae_pgd_735.json
python robust_train.py --config-robust configs_robust/mi_unet_sample_aug.json --config-attack configs_attack/mi_cvae_pgd_881.json
python robust_train.py --config-robust configs_robust/mi_unet_sample_aug.json --config-attack configs_attack/mi_cvae_pgd.json

python robust_train.py --config-robust configs_robust/mi_unet_cvae_aug.json --config-attack configs_attack/mi_clean.json
python robust_train.py --config-robust configs_robust/mi_unet_cvae_aug.json --config-attack configs_attack/mi_cvae_pgd_735.json
python robust_train.py --config-robust configs_robust/mi_unet_cvae_aug.json --config-attack configs_attack/mi_cvae_pgd_881.json
python robust_train.py --config-robust configs_robust/mi_unet_cvae_aug.json --config-attack configs_attack/mi_cvae_pgd.json

python robust_train.py --config-robust configs_robust/mi_unet_cvae_attack.json --config-attack configs_attack/mi_clean.json
python robust_train.py --config-robust configs_robust/mi_unet_cvae_attack.json --config-attack configs_attack/mi_cvae_pgd_735.json
python robust_train.py --config-robust configs_robust/mi_unet_cvae_attack.json --config-attack configs_attack/mi_cvae_pgd_881.json
python robust_train.py --config-robust configs_robust/mi_unet_cvae_attack.json --config-attack configs_attack/mi_cvae_pgd.json

# Train and get certified radius of classifier with randomized smoothing (Figure 16 of paper)
python robust_train.py --config-robust configs_robust/mi_unet_smoothing_690.json
python robust_eval.py --config-robust configs_robust/mi_unet_smoothing_690.json --config-attack configs_attack/mi_certify_690.json
