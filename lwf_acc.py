import matplotlib.pyplot as plt
import numpy as np
Few_shot = False # if true exits after few-shot plots
kd_occe_loss = True

# Data from logs
number_of_classes_init_10_incr_10 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


if Few_shot is True: 
    acc_without_std = True
    acc_with_std = True
    acc_impr_without_std = True
    acc_impr_with_std = True

    results = {}
    improvements = {}
    gamma_0_fs_50_results = np.array([
        [49.0, 36.25, 32.6, 23.68, 20.52, 18.62, 14.99, 14.1, 11.83, 11.16],
        [51.1, 33.9, 25.13, 21.55, 18.52, 16.17, 15.16, 14.68, 12.47, 11.1],
        [63.3, 38.5, 25.57, 21.3, 18.96, 17.32, 15.61, 14.24, 12.31, 11.61]])
    results['fs_50'] = {'gamma_0': {'mean': np.mean(gamma_0_fs_50_results, axis=0), 'std': np.std(gamma_0_fs_50_results, axis=0)}}

    gamma_0_01_fs_50_results = np.array([
        [51.5, 37.45, 33.7, 25.5, 23.18, 20.47, 17.23, 15.1, 12.91, 11.91],
        [52.2, 32.0, 24.47, 21.52, 18.22, 16.48, 15.19, 14.76, 12.36, 11.19],
        [62.4, 40.55, 27.67, 22.72, 20.4, 18.27, 17.47, 15.24, 13.89, 11.84]])
    results['fs_50']['gamma_0_01'] = {'mean': np.mean(gamma_0_01_fs_50_results, axis=0), 'std': np.std(gamma_0_01_fs_50_results, axis=0)}
    _dif_from_base = gamma_0_01_fs_50_results - gamma_0_fs_50_results
    improvements['fs_50'] = {'gamma_0_01': {'mean': np.mean(_dif_from_base, axis=0), 'std': np.std(_dif_from_base, axis=0)}}

    gamma_0_05_fs_50_results = np.array([
        [49.3, 38.2, 34.3, 26.28, 21.82, 19.7, 16.77, 14.75, 12.6, 11.69],
        [54.5, 36.3, 27.4, 22.28, 19.62, 17.12, 16.24, 15.56, 12.74, 11.68],
        [65.5, 43.25, 28.03, 23.18, 21.68, 20.0, 17.73, 16.2, 14.09, 13.08]])
    results['fs_50']['gamma_0_05'] = {'mean': np.mean(gamma_0_05_fs_50_results, axis=0), 'std': np.std(gamma_0_05_fs_50_results, axis=0)}
    _dif_from_base = gamma_0_05_fs_50_results - gamma_0_fs_50_results
    improvements['fs_50']['gamma_0_05'] = {'mean': np.mean(_dif_from_base, axis=0), 'std': np.std(_dif_from_base, axis=0)}

    gamma_0_1_fs_50_results = np.array([
        [52.1, 38.15, 32.93, 26.52, 20.9, 19.07, 15.39, 14.16, 11.94, 11.0],
        [54.3, 36.6, 26.53, 22.98, 19.16, 16.08, 15.6, 14.12, 12.11, 10.73],
        [66.0, 43.95, 30.4, 23.7, 21.72, 19.43, 17.5, 16.11, 12.74, 12.2]])
    results['fs_50']['gamma_0_1'] = {'mean': np.mean(gamma_0_1_fs_50_results, axis=0), 'std': np.std(gamma_0_1_fs_50_results, axis=0)}
    _dif_from_base = gamma_0_1_fs_50_results - gamma_0_fs_50_results
    improvements['fs_50']['gamma_0_1'] = {'mean': np.mean(_dif_from_base, axis=0), 'std': np.std(_dif_from_base, axis=0)}

    gamma_0_5_fs_50_results = np.array([
        [49.2, 37.05, 32.97, 24.35, 18.02, 16.45, 13.34, 11.88, 9.82, 8.81],
        [52.9, 34.65, 24.8, 22.62, 17.88, 13.6, 13.27, 11.78, 10.58, 9.23],
        [64.2, 45.95, 29.23, 23.22, 20.4, 16.93, 14.74, 13.35, 11.07, 10.8]])
    results['fs_50']['gamma_0_5'] = {'mean': np.mean(gamma_0_5_fs_50_results, axis=0), 'std': np.std(gamma_0_5_fs_50_results, axis=0)}
    _dif_from_base = gamma_0_5_fs_50_results - gamma_0_fs_50_results
    improvements['fs_50']['gamma_0_5'] = {'mean': np.mean(_dif_from_base, axis=0), 'std': np.std(_dif_from_base, axis=0)}

    gamma_0_fs_100_results = np.array([
        [59.0, 43.3, 37.3, 29.98, 25.66, 23.4, 19.89, 17.85, 14.34, 13.76],
        [61.4, 43.35, 31.87, 26.52, 23.0, 20.52, 18.37, 17.66, 15.63, 14.31],
        [73.5, 52.2, 35.23, 27.5, 25.42, 22.3, 20.24, 18.98, 17.31, 16.22]])
    results['fs_100'] = {'gamma_0': {'mean': np.mean(gamma_0_fs_100_results, axis=0), 'std': np.std(gamma_0_fs_100_results, axis=0)}}

    gamma_0_01_fs_100_results = np.array([
        [58.2, 43.85, 38.5, 31.68, 27.04, 24.93, 21.59, 19.98, 16.0, 14.6],
        [61.4, 41.65, 31.0, 25.32, 22.06, 19.45, 18.34, 17.51, 16.04, 14.87],
        [75.0, 53.55, 36.3, 29.85, 27.66, 24.6, 20.81, 19.38, 17.93, 17.5]])
    results['fs_100']['gamma_0_01'] = {'mean': np.mean(gamma_0_01_fs_100_results, axis=0), 'std': np.std(gamma_0_01_fs_100_results, axis=0)}
    _dif_from_base = gamma_0_01_fs_100_results - gamma_0_fs_100_results
    improvements['fs_100'] = {'gamma_0_01': {'mean': np.mean(_dif_from_base, axis=0), 'std': np.std(_dif_from_base, axis=0)}}

    gamma_0_05_fs_100_results = np.array([
        [60.1, 45.8, 40.27, 33.25, 28.28, 25.33, 21.21, 18.84, 16.02, 14.33],
        [62.0, 48.45, 36.07, 30.75, 25.98, 23.48, 21.13, 19.81, 17.29, 16.1],
        [72.3, 53.55, 37.77, 30.4, 29.0, 25.23, 22.04, 19.96, 17.82, 16.56]])
    results['fs_100']['gamma_0_05'] = {'mean': np.mean(gamma_0_05_fs_100_results, axis=0), 'std': np.std(gamma_0_05_fs_100_results, axis=0)}
    _dif_from_base = gamma_0_05_fs_100_results - gamma_0_fs_100_results
    improvements['fs_100']['gamma_0_05'] = {'mean': np.mean(_dif_from_base, axis=0), 'std': np.std(_dif_from_base, axis=0)}

    gamma_0_1_fs_100_results = np.array([
        [60.0, 47.45, 40.73, 33.5, 27.8, 24.72, 20.71, 19.44, 16.12, 14.58],
        [64.4, 45.95, 32.47, 30.0, 26.28, 21.87, 20.24, 19.04, 17.12, 15.03],
        [74.3, 56.2, 40.93, 30.55, 29.34, 25.18, 22.51, 20.45, 18.34, 16.63]])
    results['fs_100']['gamma_0_1'] = {'mean': np.mean(gamma_0_1_fs_100_results, axis=0), 'std': np.std(gamma_0_1_fs_100_results, axis=0)}
    _dif_from_base = gamma_0_1_fs_100_results - gamma_0_fs_100_results
    improvements['fs_100']['gamma_0_1'] = {'mean': np.mean(_dif_from_base, axis=0), 'std': np.std(_dif_from_base, axis=0)}

    gamma_0_5_fs_100_results = np.array([
        [57.3, 45.95, 40.2, 29.05, 24.52, 21.65, 19.27, 17.56, 14.22, 12.22],
        [62.4, 46.45, 35.07, 29.1, 24.36, 21.17, 18.69, 15.72, 15.09, 13.54],
        [71.3, 53.9, 37.47, 28.08, 25.68, 22.17, 19.19, 15.69, 14.06, 13.18]])
    results['fs_100']['gamma_0_5'] = {'mean': np.mean(gamma_0_5_fs_100_results, axis=0), 'std': np.std(gamma_0_5_fs_100_results, axis=0)}
    _dif_from_base = gamma_0_5_fs_100_results - gamma_0_fs_100_results
    improvements['fs_100']['gamma_0_5'] = {'mean': np.mean(_dif_from_base, axis=0), 'std': np.std(_dif_from_base, axis=0)}

    gamma_0_fs_250 = np.array([
        [65.3, 52.4, 46.47, 37.12, 32.9, 27.6, 24.59, 21.02, 18.29, 16.61],
        [71.2, 56.0, 44.1, 35.83, 30.06, 26.33, 24.06, 23.11, 21.79, 18.51],
        [84.4, 65.9, 45.3, 35.5, 30.44, 28.9, 25.66, 24.55, 22.43, 20.77]])
    results['fs_250'] = {'gamma_0': {'mean': np.mean(gamma_0_fs_250, axis=0), 'std': np.std(gamma_0_fs_250, axis=0)}}

    gamma_0_01_fs_250 = np.array([
        [66.1, 55.2, 48.3, 40.3, 34.16, 29.3, 25.43, 22.95, 19.19, 17.46],
        [70.5, 56.9, 43.43, 36.92, 31.64, 27.3, 26.11, 24.99, 23.21, 20.7],
        [82.9, 65.35, 45.17, 36.25, 34.34, 31.33, 27.24, 24.3, 22.22, 21.37]])
    results['fs_250']['gamma_0_01'] = {'mean': np.mean(gamma_0_01_fs_250, axis=0), 'std': np.std(gamma_0_01_fs_250, axis=0)}
    _dif_from_base = gamma_0_01_fs_250 - gamma_0_fs_250
    improvements['fs_250'] = {'gamma_0_01': {'mean': np.mean(_dif_from_base, axis=0), 'std': np.std(_dif_from_base, axis=0)}}

    gamma_0_05_fs_250 = np.array([
        [67.9, 55.35, 51.27, 41.22, 36.66, 32.08, 28.24, 25.14, 21.82, 19.5],
        [71.3, 57.0, 45.83, 38.25, 32.74, 28.08, 26.06, 23.98, 22.24, 20.47],
        [83.6, 65.0, 46.23, 38.92, 33.66, 31.28, 27.57, 25.35, 23.27, 20.64]])
    results['fs_250']['gamma_0_05'] = {'mean': np.mean(gamma_0_05_fs_250, axis=0), 'std': np.std(gamma_0_05_fs_250, axis=0)}
    _dif_from_base = gamma_0_05_fs_250 - gamma_0_fs_250
    improvements['fs_250']['gamma_0_05'] = {'mean': np.mean(_dif_from_base, axis=0), 'std': np.std(_dif_from_base, axis=0)}

    gamma_0_1_fs_250 = np.array([
        [67.6, 56.8, 50.33, 39.6, 35.82, 31.55, 27.09, 25.96, 21.38, 19.02],
        [72.1, 60.05, 47.9, 41.58, 34.78, 30.08, 29.34, 25.94, 23.23, 20.32],
        [85.3, 68.6, 48.67, 39.17, 35.24, 31.87, 27.23, 24.84, 21.93, 20.41]])
    results['fs_250']['gamma_0_1'] = {'mean': np.mean(gamma_0_1_fs_250, axis=0), 'std': np.std(gamma_0_1_fs_250, axis=0)}
    _dif_from_base = gamma_0_1_fs_250 - gamma_0_fs_250
    improvements['fs_250']['gamma_0_1'] = {'mean': np.mean(_dif_from_base, axis=0), 'std': np.std(_dif_from_base, axis=0)}

    gamma_0_5_fs_250 = np.array([
        [66.3, 53.95, 49.7, 38.55, 32.24, 27.63, 24.61, 21.79, 18.6, 16.44],
        [73.1, 59.1, 45.67, 38.38, 32.52, 27.78, 25.79, 22.75, 20.78, 18.37],
        [85.6, 68.5, 49.93, 39.3, 34.6, 31.17, 26.47, 23.3, 20.56, 18.54]])
    results['fs_250']['gamma_0_5'] = {'mean': np.mean(gamma_0_5_fs_250, axis=0), 'std': np.std(gamma_0_5_fs_250, axis=0)}
    _dif_from_base = gamma_0_5_fs_250 - gamma_0_fs_250
    improvements['fs_250']['gamma_0_5'] = {'mean': np.mean(_dif_from_base, axis=0), 'std': np.std(_dif_from_base, axis=0)}


    if acc_without_std:
        # Accuracy curves for Few-shot
        fig, axis = plt.subplots(1, 3, figsize=(22, 11))

        # Few-shot 10% (50 samples per class)
        axis[0].plot(number_of_classes_init_10_incr_10, results['fs_50']['gamma_0']['mean'], 'o-', label='ce', color="#1f77b4")
        axis[0].plot(number_of_classes_init_10_incr_10, results['fs_50']['gamma_0_01']['mean'], 's--', label='g = 0.01', color="#ff7f0e")
        axis[0].plot(number_of_classes_init_10_incr_10, results['fs_50']['gamma_0_05']['mean'], '^--', label='g = 0.05', color="#d62728")
        axis[0].plot(number_of_classes_init_10_incr_10, results['fs_50']['gamma_0_1']['mean'], 'x--', label='g = 0.1', color="#9467bd")
        axis[0].plot(number_of_classes_init_10_incr_10, results['fs_50']['gamma_0_5']['mean'], 'o-', label='g = 0.5', color="#e377c2")
        axis[0].set_xlabel('Number of Classes')
        axis[0].set_ylabel('Accuracy (%)')
        axis[0].grid(True, linestyle="--", alpha=0.6)
        axis[0].legend()
        axis[0].set_title('Few-shot 10% (50 samples per class)')

        # Few-shot 20% (100 samples per class)
        axis[1].plot(number_of_classes_init_10_incr_10, results['fs_100']['gamma_0']['mean'], 'o-', label='ce', color="#1f77b4")
        axis[1].plot(number_of_classes_init_10_incr_10, results['fs_100']['gamma_0_01']['mean'], 's--', label='g = 0.01', color="#ff7f0e")
        axis[1].plot(number_of_classes_init_10_incr_10, results['fs_100']['gamma_0_05']['mean'], '^--', label='g = 0.05', color="#d62728")
        axis[1].plot(number_of_classes_init_10_incr_10, results['fs_100']['gamma_0_1']['mean'], 'x--', label='g = 0.1', color="#9467bd")
        axis[1].plot(number_of_classes_init_10_incr_10, results['fs_100']['gamma_0_5']['mean'], 'o-', label='g = 0.5', color="#e377c2")
        axis[1].set_xlabel('Number of Classes')
        axis[1].set_ylabel('Accuracy (%)')
        axis[1].grid(True, linestyle="--", alpha=0.6)
        axis[1].legend()
        axis[1].set_title('Few-shot 10% (50 samples per class)')

        # Few-shot 50% (250 samples per class)
        axis[2].plot(number_of_classes_init_10_incr_10, results['fs_250']['gamma_0']['mean'], 'o--',  label='g = 0', color="#1f77b4")
        axis[2].plot(number_of_classes_init_10_incr_10, results['fs_250']['gamma_0_01']['mean'], 'o-', label='g = 0.01', color="#ff7f0e")
        axis[2].plot(number_of_classes_init_10_incr_10, results['fs_250']['gamma_0_05']['mean'], 's-', label='g = 0.05', color="#d62728")
        axis[2].plot(number_of_classes_init_10_incr_10, results['fs_250']['gamma_0_1']['mean'], '^-', label='g = 0.1', color="#9467bd")
        axis[2].plot(number_of_classes_init_10_incr_10, results['fs_250']['gamma_0_5']['mean'], '.-', label='g = 0.5', color="#e377c2")
        axis[2].set_xlabel('Number of Classes')
        axis[2].set_ylabel('Accuracy (%)')
        axis[2].grid(True, linestyle="--", alpha=0.6)
        axis[2].legend()
        axis[2].set_title('Few-shot 50% (250 samples per class)')

        plt.suptitle("LwF Few-Shot OCCE Accuracies", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig("resources/lwf_accuracies_for_few_shot.png")
        plt.close()

    if acc_with_std:
        # Plot acc curves for Few-shot with std
        fig, axis = plt.subplots(1, 3, figsize=(35, 13))

        # Few-shot 10% (50 samples per class)
        axis[0].errorbar(number_of_classes_init_10_incr_10, results['fs_50']['gamma_0']['mean'], yerr=results['fs_50']['gamma_0']['std'], fmt='o--', capsize=8, label='g = 0', color="#1f77b4")
        axis[0].errorbar(number_of_classes_init_10_incr_10, results['fs_50']['gamma_0_01']['mean'], yerr=results['fs_50']['gamma_0_01']['std'], fmt='o-', capsize=8, label='g = 0.01', color="#ff7f0e")
        axis[0].errorbar(number_of_classes_init_10_incr_10, results['fs_50']['gamma_0_05']['mean'], yerr=results['fs_50']['gamma_0_05']['std'], fmt='s-', capsize=8, label='g = 0.05', color="#d62728")
        axis[0].errorbar(number_of_classes_init_10_incr_10, results['fs_50']['gamma_0_1']['mean'], yerr=results['fs_50']['gamma_0_1']['std'], fmt='^-', capsize=8, label='g = 0.1', color="#9467bd")
        axis[0].errorbar(number_of_classes_init_10_incr_10, results['fs_50']['gamma_0_5']['mean'], yerr=results['fs_50']['gamma_0_5']['std'], fmt='.-', capsize=8, label='g = 0.5', color="#e377c2")
        axis[0].set_xlabel('Number of Classes')
        axis[0].set_ylabel('Accuracy (%)')
        axis[0].grid(True, linestyle="--", alpha=0.6)
        axis[0].legend()
        axis[0].set_title('Few-shot 10% (50 samples per class)')

        # Few-shot 20% (100 samples per class)
        axis[1].errorbar(number_of_classes_init_10_incr_10, results['fs_100']['gamma_0']['mean'], yerr=results['fs_100']['gamma_0']['std'], fmt='o--', capsize=8, label='g = 0', color="#1f77b4")
        axis[1].errorbar(number_of_classes_init_10_incr_10, results['fs_100']['gamma_0_01']['mean'], yerr=results['fs_100']['gamma_0_01']['std'], fmt='o-', capsize=8, label='g = 0.01', color="#ff7f0e")
        axis[1].errorbar(number_of_classes_init_10_incr_10, results['fs_100']['gamma_0_05']['mean'], yerr=results['fs_100']['gamma_0_05']['std'], fmt='s-', capsize=8, label='g = 0.05', color="#d62728")
        axis[1].errorbar(number_of_classes_init_10_incr_10, results['fs_100']['gamma_0_1']['mean'], yerr=results['fs_100']['gamma_0_1']['std'], fmt='^-', capsize=8, label='g = 0.1', color="#9467bd")
        axis[1].errorbar(number_of_classes_init_10_incr_10, results['fs_100']['gamma_0_5']['mean'], yerr=results['fs_100']['gamma_0_5']['std'], fmt='.-', capsize=8, label='g = 0.5', color="#e377c2")
        axis[1].set_xlabel('Number of Classes')
        axis[1].set_ylabel('Accuracy (%)')
        axis[1].grid(True, linestyle="--", alpha=0.6)
        axis[1].legend()
        axis[1].set_title('Few-shot 20% (100 samples per class)')

        # Few-shot 50% (250 samples per class)
        axis[2].errorbar(number_of_classes_init_10_incr_10, results['fs_250']['gamma_0']['mean'], yerr=results['fs_250']['gamma_0']['std'], fmt='o--', capsize=8, label='g = 0', color="#1f77b4")
        axis[2].errorbar(number_of_classes_init_10_incr_10, results['fs_250']['gamma_0_01']['mean'], yerr=results['fs_250']['gamma_0_01']['std'], fmt='o-', capsize=8, label='g = 0.01', color="#ff7f0e")
        axis[2].errorbar(number_of_classes_init_10_incr_10, results['fs_250']['gamma_0_05']['mean'], yerr=results['fs_250']['gamma_0_05']['std'], fmt='s-', capsize=8, label='g = 0.05', color="#d62728")
        axis[2].errorbar(number_of_classes_init_10_incr_10, results['fs_250']['gamma_0_1']['mean'], yerr=results['fs_250']['gamma_0_1']['std'], fmt='^-', capsize=8, label='g = 0.1', color="#9467bd")
        axis[2].errorbar(number_of_classes_init_10_incr_10, results['fs_250']['gamma_0_5']['mean'], yerr=results['fs_250']['gamma_0_5']['std'], fmt='.-', capsize=8, label='g = 0.5', color="#e377c2")
        axis[2].set_xlabel('Number of Classes')
        axis[2].set_ylabel('Accuracy (%)')
        axis[2].grid(True, linestyle="--", alpha=0.6)
        axis[2].legend()
        axis[2].set_title('Few-shot 50% (250 samples per class)')

        plt.suptitle("LwF Few-Shot OCCE Accuracies +/- std", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig("resources/lwf_accuracies_with_std_for_few_shot.png")
        plt.close()
    
    if acc_impr_without_std:
        # Improvement curves for Few-shot
        fig, axis = plt.subplots(1, 3, figsize=(22, 11))

        # Few-shot 10% (50 samples per class)
        axis[0].plot(number_of_classes_init_10_incr_10, improvements['fs_50']['gamma_0_01']['mean'], 'o-', label='g = 0.01', color="#ff7f0e")
        axis[0].plot(number_of_classes_init_10_incr_10, improvements['fs_50']['gamma_0_05']['mean'], 's--', label='g = 0.05', color="#d62728")
        axis[0].plot(number_of_classes_init_10_incr_10, improvements['fs_50']['gamma_0_1']['mean'], '^--', label='g = 0.1', color="#9467bd")
        axis[0].plot(number_of_classes_init_10_incr_10, improvements['fs_50']['gamma_0_5']['mean'], 'o-', label='g = 0.5', color="#e377c2")
        axis[0].set_xlabel('Number of Classes')
        axis[0].set_ylabel('Accuracy Improvement (%)')
        axis[0].grid(True, linestyle="--", alpha=0.6)
        axis[0].legend()
        axis[0].set_title('Few-shot 10% (50 samples per class)')
        # Few-shot 20% (100 samples per class)
        axis[1].plot(number_of_classes_init_10_incr_10, improvements['fs_100']['gamma_0_01']['mean'], 'o-', label='g = 0.01', color="#ff7f0e")
        axis[1].plot(number_of_classes_init_10_incr_10, improvements['fs_100']['gamma_0_05']['mean'], 's--', label='g = 0.05', color="#d62728")
        axis[1].plot(number_of_classes_init_10_incr_10, improvements['fs_100']['gamma_0_1']['mean'], '^--', label='g = 0.1', color="#9467bd")
        axis[1].plot(number_of_classes_init_10_incr_10, improvements['fs_100']['gamma_0_5']['mean'], 'o-', label='g = 0.5', color="#e377c2")
        axis[1].set_xlabel('Number of Classes')
        axis[1].set_ylabel('Accuracy Improvement (%)')
        axis[1].grid(True, linestyle="--", alpha=0.6)
        axis[1].legend()
        axis[1].set_title('Few-shot 20% (100 samples per class)')
        # Few-shot 50% (250 samples per class)
        axis[2].plot(number_of_classes_init_10_incr_10, improvements['fs_250']['gamma_0_01']['mean'], 'o--', label='g = 0.01', color="#ff7f0e")
        axis[2].plot(number_of_classes_init_10_incr_10, improvements['fs_250']['gamma_0_05']['mean'], 's-', label='g = 0.05', color="#d62728")
        axis[2].plot(number_of_classes_init_10_incr_10, improvements['fs_250']['gamma_0_1']['mean'], '^-', label='g = 0.1', color="#9467bd")
        axis[2].plot(number_of_classes_init_10_incr_10, improvements['fs_250']['gamma_0_5']['mean'], '.-', label='g = 0.5', color="#e377c2")
        axis[2].set_xlabel('Number of Classes')
        axis[2].set_ylabel('Accuracy Improvement (%)')
        axis[2].grid(True, linestyle="--", alpha=0.6)
        axis[2].legend()
        axis[2].set_title('Few-shot 50% (250 samples per class)')

        plt.suptitle("LwF Few-Shot OCCE Accuracy Improvements from CE", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig("resources/lwf_accuracy_improvements_from_ce_for_few_shot.png")
        plt.close()

    if acc_impr_with_std : 
        # Improvement curves for Few-shot
        fig, axis = plt.subplots(1, 3, figsize=(35, 13))

        # Few-shot 10% (50 samples per class)
        axis[0].errorbar(number_of_classes_init_10_incr_10, improvements['fs_50']['gamma_0_01']['mean'], yerr=improvements['fs_50']['gamma_0_01']['std'], fmt='o-', capsize=6, label='g = 0.01', color="#ff7f0e")
        axis[0].errorbar(number_of_classes_init_10_incr_10, improvements['fs_50']['gamma_0_05']['mean'], yerr=improvements['fs_50']['gamma_0_05']['std'], fmt='s--', capsize=6, label='g = 0.05', color="#d62728")
        axis[0].errorbar(number_of_classes_init_10_incr_10, improvements['fs_50']['gamma_0_1']['mean'], yerr=improvements['fs_50']['gamma_0_1']['std'], fmt='^--', capsize=6, label='g = 0.1', color="#9467bd")
        axis[0].errorbar(number_of_classes_init_10_incr_10, improvements['fs_50']['gamma_0_5']['mean'], yerr=improvements['fs_50']['gamma_0_5']['std'], fmt='o-', capsize=6, label='g = 0.5', color="#e377c2")
        axis[0].set_xlabel('Number of Classes')
        axis[0].set_ylabel('Accuracy Improvement (%)')
        axis[0].grid(True, linestyle="--", alpha=0.6)
        axis[0].legend()
        axis[0].set_title('Few-shot 10% (50 samples per class)')

        # Few-shot 20% (100 samples per class)
        axis[1].errorbar(number_of_classes_init_10_incr_10, improvements['fs_100']['gamma_0_01']['mean'], yerr=improvements['fs_100']['gamma_0_01']['std'], fmt='o-', capsize=6, label='g = 0.01', color="#ff7f0e")
        axis[1].errorbar(number_of_classes_init_10_incr_10, improvements['fs_100']['gamma_0_05']['mean'], yerr=improvements['fs_100']['gamma_0_05']['std'], fmt='s--', capsize=6, label='g = 0.05', color="#d62728")
        axis[1].errorbar(number_of_classes_init_10_incr_10, improvements['fs_100']['gamma_0_1']['mean'], yerr=improvements['fs_100']['gamma_0_1']['std'], fmt='^--', capsize=6, label='g = 0.1', color="#9467bd")
        axis[1].errorbar(number_of_classes_init_10_incr_10, improvements['fs_100']['gamma_0_5']['mean'], yerr=improvements['fs_100']['gamma_0_5']['std'], fmt='o-', capsize=6, label='g = 0.5', color="#e377c2")
        axis[1].set_xlabel('Number of Classes')
        axis[1].set_ylabel('Accuracy Improvement (%)')
        axis[1].grid(True, linestyle="--", alpha=0.6)
        axis[1].legend()
        axis[1].set_title('Few-shot 20% (100 samples per class)')

        # Few-shot 50% (250 samples per class)
        axis[2].errorbar(number_of_classes_init_10_incr_10, improvements['fs_250']['gamma_0_01']['mean'], yerr=improvements['fs_250']['gamma_0_01']['std'], fmt='o--', capsize=6, label='g = 0.01', color="#ff7f0e")
        axis[2].errorbar(number_of_classes_init_10_incr_10, improvements['fs_250']['gamma_0_05']['mean'], yerr=improvements['fs_250']['gamma_0_05']['std'], fmt='s-', capsize=6, label='g = 0.05', color="#d62728")
        axis[2].errorbar(number_of_classes_init_10_incr_10, improvements['fs_250']['gamma_0_1']['mean'], yerr=improvements['fs_250']['gamma_0_1']['std'], fmt='^--', capsize=6, label='g = 0.1', color="#9467bd")
        axis[2].errorbar(number_of_classes_init_10_incr_10, improvements['fs_250']['gamma_0_5']['mean'], yerr=improvements['fs_250']['gamma_0_5']['std'], fmt='o-', capsize=6, label='g = 0.5', color="#e377c2")
        axis[2].set_xlabel('Number of Classes')
        axis[2].set_ylabel('Accuracy Improvement (%)')
        axis[2].grid(True, linestyle="--", alpha=0.6)
        axis[2].legend()
        axis[2].set_title('Few-shot 50% (250 samples per class)')

        plt.suptitle("LwF Few-Shot OCCE Acc IMPROVEMENTS from CE +/- std", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig("resources/lwf_accuracy_improvements_from_ce_with_std_for_few_shot.png")
        plt.close()
    
    exit()
    

if kd_occe_loss : 
    kd_acc_without_std = True
    kd_acc_impr_without_std = True
    kd_acc_impr_with_std = True

    # HERE GAMMA = 0 AND WE LOOK AT THE EFFECT OF KD
    kd_gamma_0 = np.array([
        [76.3, 61.45, 53.6, 42.92, 38.7, 35.37, 30.34, 26.18, 23.63, 21.29],
        [78.2, 64.6, 50.67, 40.7, 35.52, 32.65, 26.17, 26.01, 23.64, 23.37],
        [89.4, 73.3, 56.97, 46.15, 37.48, 33.73, 29.43, 27.5, 24.83, 23.47]
    ])  # same with gamma = 0
    results = {'ce': {'mean': np.mean(kd_gamma_0, axis=0), 'std': np.std(kd_gamma_0, axis=0)}}

    kd_gamma_0_01 = np.array([
        [74.8, 60.1, 51.23, 42.8, 38.62, 34.03, 31.1, 26.22, 22.76, 20.78],
        [81.0, 67.55, 49.33, 40.83, 35.66, 32.03, 28.23, 26.61, 24.0, 23.19],
        [89.4, 72.55, 53.87, 42.58, 36.3, 33.65, 29.56, 27.24, 25.41, 24.46]
    ])
    results['kd_gamma_0_01'] = {'mean': np.mean(kd_gamma_0_01, axis=0), 'std': np.std(kd_gamma_0_01, axis=0)}
    _dif_from_base = kd_gamma_0_01 - kd_gamma_0
    improvements = {'kd_gamma_0_01': {'mean': np.mean(_dif_from_base, axis=0), 'std': np.std(_dif_from_base, axis=0)}}

    kd_gamma_0_02 = np.array([
        [74.8, 59.7, 52.03, 44.28, 39.26, 33.47, 29.99, 25.29, 22.88, 20.7],
        [81.0, 68.15, 50.77, 40.1, 34.66, 31.7, 28.34, 26.7, 24.9, 23.13],
        [89.4, 72.55, 53.73, 43.72, 35.84, 34.08, 30.41, 27.7, 25.3, 24.35]
    ])
    results['kd_gamma_0_02'] = {'mean': np.mean(kd_gamma_0_02, axis=0), 'std': np.std(kd_gamma_0_02, axis=0)}
    _dif_from_base = kd_gamma_0_02 - kd_gamma_0
    improvements['kd_gamma_0_02'] = {'mean': np.mean(_dif_from_base, axis=0), 'std': np.std(_dif_from_base, axis=0)}

    kd_gamma_0_05 = np.array([
        [74.8, 60.8, 52.7, 43.65, 38.42, 33.32, 30.96, 25.69, 22.2, 20.29] ,
        [81.0, 66.55, 50.57, 40.15, 34.72, 31.22, 27.94, 26.86, 24.82, 23.17],
        [89.4, 72.9, 53.3, 43.5, 36.98, 33.03, 30.7, 28.15, 25.26, 24.11]
    ])
    results['kd_gamma_0_05'] = {'mean': np.mean(kd_gamma_0_05, axis=0), 'std': np.std(kd_gamma_0_05, axis=0)}
    _dif_from_base = kd_gamma_0_05 - kd_gamma_0
    improvements['kd_gamma_0_05'] = {'mean': np.mean(_dif_from_base, axis=0), 'std': np.std(_dif_from_base, axis=0)}

    kd_gamma_0_1 = np.array([
        [74.8, 60.05, 53.0, 43.0, 38.94, 34.03, 31.44, 25.8, 22.24, 20.7] ,
        [81.0, 66.2, 49.27, 39.52, 34.64, 31.42, 27.81, 26.86, 25.12, 23.9],
        [89.4, 72.7, 53.73, 42.7, 35.94, 33.95, 31.41, 28.79, 26.18, 24.72]
    ])
    results['kd_gamma_0_1'] = {'mean': np.mean(kd_gamma_0_1, axis=0), 'std': np.std(kd_gamma_0_1, axis=0)}
    _dif_from_base = kd_gamma_0_1 - kd_gamma_0
    improvements['kd_gamma_0_1'] = {'mean': np.mean(_dif_from_base, axis=0), 'std': np.std(_dif_from_base, axis=0)}

    kd_gamma_0_2 = np.array([
        [74.8, 61.9, 53.0, 42.78, 38.66, 32.57, 30.63, 26.11, 22.6, 20.52],
        [81.0, 66.4, 49.23, 40.1, 34.92, 31.75, 27.43, 26.9, 24.91, 23.5],
        [89.4, 71.45, 53.9, 43.58, 36.5, 34.17, 31.5, 28.99, 26.01, 24.48]
    ])
    results['kd_gamma_0_2'] = {'mean': np.mean(kd_gamma_0_2, axis=0), 'std': np.std(kd_gamma_0_2, axis=0)}
    _dif_from_base = kd_gamma_0_2 - kd_gamma_0
    improvements['kd_gamma_0_2'] = {'mean': np.mean(_dif_from_base, axis=0), 'std': np.std(_dif_from_base, axis=0)}

    kd_gamma_0_5 = np.array([
        [74.8, 60.85, 52.57, 43.0, 38.48, 33.77, 31.16, 26.54, 23.26, 20.98],
        [81.0, 67.95, 50.03, 40.52, 35.46, 32.45, 28.69, 27.01, 25.67, 24.03],
        [89.4, 72.25, 53.37, 42.95, 36.46, 33.45, 31.39, 28.75, 25.81, 24.5]
    ])
    results['kd_gamma_0_5'] = {'mean': np.mean(kd_gamma_0_5, axis=0), 'std': np.std(kd_gamma_0_5, axis=0)}
    _dif_from_base = kd_gamma_0_5 - kd_gamma_0
    improvements['kd_gamma_0_5'] = {'mean': np.mean(_dif_from_base, axis=0), 'std': np.std(_dif_from_base, axis=0)}

    kd_gamma_1 = np.array([
        [74.8, 61.3, 53.3, 43.52, 37.7, 32.83, 30.97, 26.46, 22.44, 20.73],
        [81.0, 66.65, 49.53, 40.7, 34.76, 32.18, 28.57, 27.61, 25.62, 23.99],
        [89.4, 73.05, 54.77, 44.28, 37.7, 34.52, 31.54, 28.42, 25.79, 24.9]
    ])
    results['kd_gamma_1'] = {'mean': np.mean(kd_gamma_1, axis=0), 'std': np.std(kd_gamma_1, axis=0)}
    _dif_from_base = kd_gamma_1 - kd_gamma_0
    improvements['kd_gamma_1'] = {'mean': np.mean(_dif_from_base, axis=0), 'std': np.std(_dif_from_base, axis=0)}

    if kd_acc_without_std:
        # Plot accuracy curves
        plt.subplots(figsize=(15, 12))
        plt.plot(number_of_classes_init_10_incr_10, results['ce']['mean'], 'o--', label='ce', color="#1f77b4")
        plt.plot(number_of_classes_init_10_incr_10, results['kd_gamma_0_01']['mean'], 'o--', label='kd_g = 0.01', color="#ff7f0e")
        plt.plot(number_of_classes_init_10_incr_10, results['kd_gamma_0_02']['mean'], 'o--', label='kd_g = 0.02', color="#2ca02c")
        plt.plot(number_of_classes_init_10_incr_10, results['kd_gamma_0_05']['mean'],'o--', label='kd_g = 0.05', color="#d62728")
        plt.plot(number_of_classes_init_10_incr_10, results['kd_gamma_0_1']['mean'], 'o--', label='kd_g = 0.1', color="#9467bd")
        plt.plot(number_of_classes_init_10_incr_10, results['kd_gamma_0_2']['mean'], 'o-', label='kd_g = 0.2', color="#8c564b")
        plt.plot(number_of_classes_init_10_incr_10, results['kd_gamma_0_5']['mean'], 'o-', label='kd_g = 0.5', color="#e377c2")
        plt.plot(number_of_classes_init_10_incr_10, results['kd_gamma_1']['mean'], 'o-', label='kd_g = 1', color="#7f7f7f")
        plt.xlabel('Number of Classes')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.title('Lwf Model with OCCE KD loss for different kd_gamma (gamma=0 everywhere -> ce only in the new tasks)')

        plt.tight_layout()
        plt.savefig("resources/lwf_occe_kd_loss_accuracies.png")       
        plt.close()

    if kd_acc_impr_without_std:
        # Plot accuracy curves
        plt.subplots(figsize=(15, 12))
        plt.plot(number_of_classes_init_10_incr_10, improvements['kd_gamma_0_01']['mean'], 'o--', label='kd_g = 0.01', color="#ff7f0e")
        plt.plot(number_of_classes_init_10_incr_10, improvements['kd_gamma_0_02']['mean'], 'o--', label='kd_g = 0.02', color="#2ca02c")
        plt.plot(number_of_classes_init_10_incr_10, improvements['kd_gamma_0_05']['mean'], 'o--', label='kd_g = 0.05', color="#d62728")
        plt.plot(number_of_classes_init_10_incr_10, improvements['kd_gamma_0_1']['mean'], 'o--', label='kd_g = 0.1', color="#9467bd")
        plt.plot(number_of_classes_init_10_incr_10, improvements['kd_gamma_0_2']['mean'], 'o-', label='kd_g = 0.2', color="#8c564b")
        plt.plot(number_of_classes_init_10_incr_10, improvements['kd_gamma_0_5']['mean'], 'o-', label='kd_g = 0.5', color="#e377c2")
        plt.plot(number_of_classes_init_10_incr_10, improvements['kd_gamma_1']['mean'], 'o-', label='kd_g = 1', color="#7f7f7f")
        plt.xlabel('Number of Classes')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.title('Lwf Model with OCCE KD loss, accuracy improvements from kd_gamma=0 (gamma=0 everywhere -> ce only in the new tasks)')

        plt.tight_layout()
        plt.savefig("resources/lwf_occe_kd_loss_accuracy_improvements_from_ce.png")       
        plt.close()

    if kd_acc_impr_with_std:
        # Plot improvement curves with std
        plt.subplots(figsize=(15, 12))
        plt.errorbar(number_of_classes_init_10_incr_10, improvements["kd_gamma_0_01"]["mean"], yerr=improvements["kd_gamma_0_01"]["std"], fmt='o--', capsize=9, label='kd_g = 0.01', color="#2ca02c")
        plt.errorbar(number_of_classes_init_10_incr_10, improvements["kd_gamma_0_02"]["mean"], yerr=improvements["kd_gamma_0_02"]["std"], fmt='o--', capsize=8, label='kd_g = 0.02', color="#d62728")
        plt.errorbar(number_of_classes_init_10_incr_10, improvements["kd_gamma_0_05"]["mean"], yerr=improvements["kd_gamma_0_05"]["std"], fmt='o--', capsize=7, label='kd_g = 0.05', color="#ff7f0e")
        plt.errorbar(number_of_classes_init_10_incr_10, improvements["kd_gamma_0_1"]["mean"], yerr=improvements["kd_gamma_0_1"]["std"], fmt='o--', capsize=6, label='kd_g = 0.1', color="#9467bd")
        plt.errorbar(number_of_classes_init_10_incr_10, improvements["kd_gamma_0_2"]["mean"], yerr=improvements["kd_gamma_0_2"]["std"], fmt='o-', capsize=9, label='kd_g = 0.2', color="#8c564b")
        plt.errorbar(number_of_classes_init_10_incr_10, improvements["kd_gamma_0_5"]["mean"], yerr=improvements["kd_gamma_0_5"]["std"], fmt='o-', capsize=8, label='kd_g = 0.5', color="#e377c2")
        plt.errorbar(number_of_classes_init_10_incr_10, improvements["kd_gamma_1"]["mean"], yerr=improvements["kd_gamma_1"]["std"], fmt='o-', capsize=6, label='kd_g = 1', color="#7f7f7f")
        plt.xlabel('Number of Classes')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.title('Lwf Model with OCCE KD loss, accuracy improvements from kd_gamma=0 with +/- std (gamma=0 everywhere -> ce only in the new tasks)')

        plt.tight_layout()
        plt.savefig("resources/lwf_occe_kd_loss_accuracy_improvements_from_ce_with_std.png")



        exit()

# Accuracy of Lwf model with different gamma and seeds for base 10 incr 10 
# - trained for 200 epochs for base task and 150 epochs for incr

acc_without_std = True
acc_with_std = True
acc_impr_without_std = True
acc_impr_with_std = True

gamma_0 = np.array([
    [76.3, 61.45, 53.6, 42.92, 38.7, 35.37, 30.34, 26.18, 23.63, 21.29],
    [78.2, 64.6, 50.67, 40.7, 35.52, 32.65, 26.17, 26.01, 23.64, 23.37],
    [89.4, 73.3, 56.97, 46.15, 37.48, 33.73, 29.43, 27.5, 24.83, 23.47]
])
results = {'ce': {'mean': np.mean(gamma_0, axis=0), 'std': np.std(gamma_0, axis=0)}}

gamma_0_01 = np.array([
    [75.3, 62.4, 54.93, 45.52, 41.38, 36.9, 32.09, 29.0, 25.5, 22.46],
    [81.2, 68.75, 53.2, 45.48, 39.04, 35.28, 30.5, 28.92, 26.34, 24.97],
    [90.4, 75.6, 59.77, 47.48, 42.06, 37.93, 31.89, 28.86, 25.9, 25.08]
])
results['gamma_0_01'] = {'mean': np.mean(gamma_0_01, axis=0), 'std': np.std(gamma_0_01, axis=0)}
_dif_from_base = gamma_0_01 - gamma_0
improvements = {'gamma_0_01': {'mean': np.mean(_dif_from_base, axis=0), 'std': np.std(_dif_from_base, axis=0)}}

gamma_0_02 = np.array([
    [77.6, 62.9, 57.8, 47.2, 43.5, 38.1, 33.09, 29.84, 25.86, 22.13],
    [80.7, 68.85, 54.77, 45.88, 39.62, 34.02, 30.24, 29.05, 25.7, 23.55],
    [89.5, 75.5, 57.5, 47.12, 41.66, 37.7, 31.7, 29.56, 26.97, 25.35]
])
results['gamma_0_02'] = {'mean': np.mean(gamma_0_02, axis=0), 'std': np.std(gamma_0_02, axis=0)}
_dif_from_base = gamma_0_02 - gamma_0
improvements['gamma_0_02'] = {'mean': np.mean(_dif_from_base, axis=0), 'std': np.std(_dif_from_base, axis=0)}

gamma_0_05 = np.array([
    [78.2, 63.4, 58.67, 46.92, 42.46, 36.57, 32.53, 30.04, 26.08, 22.14],
    [82.0, 69.15, 55.23, 46.42, 40.44, 33.32, 30.03, 28.5, 24.88, 22.67],
    [89.4, 78.25, 59.8, 49.62, 43.02, 37.33, 32.0, 29.28, 26.16, 23.11]
])
results['gamma_0_05'] = {'mean': np.mean(gamma_0_05, axis=0), 'std': np.std(gamma_0_05, axis=0)}
_dif_from_base = gamma_0_05 - gamma_0
improvements['gamma_0_05'] = {'mean': np.mean(_dif_from_base, axis=0), 'std': np.std(_dif_from_base, axis=0)}

gamma_0_1 = np.array([
    [74.9, 61.25, 56.2, 46.82, 42.3, 35.8, 31.07, 28.65, 25.09, 21.43],
    [80.5, 69.15, 53.77, 46.95, 40.4, 33.77, 29.83, 27.44, 24.01, 21.91],
    [89.2, 77.4, 59.7, 49.32, 42.96, 37.2, 31.43, 27.65, 24.48, 22.36]
])
results['gamma_0_1'] = {'mean': np.mean(gamma_0_1, axis=0), 'std': np.std(gamma_0_1, axis=0)}
_dif_from_base = gamma_0_1 - gamma_0
improvements['gamma_0_1'] = {'mean': np.mean(_dif_from_base, axis=0), 'std': np.std(_dif_from_base, axis=0)}

gamma_0_2 = np.array([
    [75.8, 62.25, 56.63, 45.45, 40.94, 34.35, 30.27, 27.25, 23.51, 20.09],
    [82.2, 70.5, 55.23, 46.82, 40.24, 31.27, 29.63, 27.0, 24.1, 22.29],
    [90.6, 78.4, 60.1, 48.62, 42.46, 37.12, 29.46, 24.85, 23.21, 21.57]
])
results['gamma_0_2'] = {'mean': np.mean(gamma_0_2, axis=0), 'std': np.std(gamma_0_2, axis=0)}
_dif_from_base = gamma_0_2 - gamma_0
improvements['gamma_0_2'] = {'mean': np.mean(_dif_from_base, axis=0), 'std': np.std(_dif_from_base, axis=0)}

gamma_0_5 = np.array([
    [75.9, 62.3, 57.63, 45.22, 41.62, 35.02, 30.26, 26.59, 24.74, 20.65],
    [78.1, 68.1, 52.77, 45.5, 39.02, 31.97, 28.3, 25.84, 23.02, 20.28],
    [90.0, 78.05, 59.47, 49.0, 40.8, 35.12, 29.44, 25.85, 22.9, 20.79]
])
results['gamma_0_5'] = {'mean': np.mean(gamma_0_5, axis=0), 'std': np.std(gamma_0_5, axis=0)}
_dif_from_base = gamma_0_5 - gamma_0
improvements['gamma_0_5'] = {'mean': np.mean(_dif_from_base, axis=0), 'std': np.std(_dif_from_base, axis=0)}

gamma_1 = np.array([
    [76.1, 61.55, 55.77, 44.18, 39.7, 33.55, 29.41, 26.44, 23.12, 19.34],
    [81.9, 68.8, 52.83, 45.68, 38.04, 31.57, 28.67, 26.3, 23.68, 21.02],
    [90.1, 77.6, 57.9, 47.48, 40.0, 34.38, 27.87, 25.35, 22.79, 20.57]
])
results['gamma_1'] = {'mean': np.mean(gamma_1, axis=0), 'std': np.std(gamma_1, axis=0)}
_dif_from_base = gamma_1 - gamma_0
improvements['gamma_1'] = {'mean': np.mean(_dif_from_base, axis=0), 'std': np.std(_dif_from_base, axis=0)}

if acc_without_std:
    # Plot accuracy curves
    plt.subplots(figsize=(15, 12))
    plt.plot(number_of_classes_init_10_incr_10, results['ce']['mean'], 'o--', label='ce', color="#1f77b4")
    plt.plot(number_of_classes_init_10_incr_10, results['gamma_0_01']['mean'], 'o--', label='g = 0.01', color="#ff7f0e")
    plt.plot(number_of_classes_init_10_incr_10, results['gamma_0_02']['mean'], 'o--', label='g = 0.02', color="#2ca02c")
    plt.plot(number_of_classes_init_10_incr_10, results['gamma_0_05']['mean'],'o--', label='g = 0.05', color="#d62728")
    plt.plot(number_of_classes_init_10_incr_10, results['gamma_0_1']['mean'], 'o--', label='g = 0.1', color="#9467bd")
    plt.plot(number_of_classes_init_10_incr_10, results['gamma_0_2']['mean'], 'o-', label='g = 0.2', color="#8c564b")
    plt.plot(number_of_classes_init_10_incr_10, results['gamma_0_5']['mean'], 'o-', label='g = 0.5', color="#e377c2")
    plt.plot(number_of_classes_init_10_incr_10, results['gamma_1']['mean'], 'o-', label='g = 1', color="#7f7f7f")
    plt.xlabel('Number of Classes')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Lwf Model mean Accuracies')

    plt.tight_layout()
    plt.savefig("resources/lwf_accuracies.png")       
    plt.close()

if acc_with_std:
    # Plot accuracy curves with std
    plt.subplots(figsize=(15, 12))
    plt.errorbar(number_of_classes_init_10_incr_10, results["ce"]["mean"], yerr=results["ce"]["std"], fmt='o--', capsize=8, label='ce', color="#1f77b4")
    plt.errorbar(number_of_classes_init_10_incr_10, results["gamma_0_01"]["mean"], yerr=results["gamma_0_01"]["std"], fmt='o--', capsize=8, label='g = 0.01', color="#2ca02c")
    plt.errorbar(number_of_classes_init_10_incr_10, results["gamma_0_02"]["mean"], yerr=results["gamma_0_02"]["std"], fmt='o--', capsize=8, label='g = 0.02', color="#d62728")
    plt.errorbar(number_of_classes_init_10_incr_10, results["gamma_0_05"]["mean"], yerr=results["gamma_0_05"]["std"], fmt='o--', capsize=8, label='g = 0.05', color="#ff7f0e")
    plt.errorbar(number_of_classes_init_10_incr_10, results["gamma_0_1"]["mean"], yerr=results["gamma_0_1"]["std"], fmt='o--', capsize=8, label='g = 0.1', color="#9467bd")
    plt.errorbar(number_of_classes_init_10_incr_10, results["gamma_0_2"]["mean"], yerr=results["gamma_0_2"]["std"], fmt='o-', capsize=8, label='g = 0.2', color="#8c564b")
    plt.errorbar(number_of_classes_init_10_incr_10, results["gamma_0_5"]["mean"], yerr=results["gamma_0_5"]["std"], fmt='o-', capsize=8, label='g = 0.5', color="#e377c2")
    plt.errorbar(number_of_classes_init_10_incr_10, results["gamma_1"]["mean"], yerr=results["gamma_1"]["std"], fmt='o-', capsize=8, label='g = 1', color="#7f7f7f")
    plt.xlabel('Number of Classes')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Lwf Model mean Accuracies +/- std')

    plt.tight_layout()
    plt.savefig("resources/lwf_accuracies_with_std.png")       
    plt.close()

if acc_impr_without_std:
    # Plot accuracy curves
    plt.subplots(figsize=(15, 12))
    plt.plot(number_of_classes_init_10_incr_10, improvements['gamma_0_01']['mean'], 'o--', label='g = 0.01', color="#ff7f0e")
    plt.plot(number_of_classes_init_10_incr_10, improvements['gamma_0_02']['mean'], 'o--', label='g = 0.02', color="#2ca02c")
    plt.plot(number_of_classes_init_10_incr_10, improvements['gamma_0_05']['mean'], 'o--', label='g = 0.05', color="#d62728")
    plt.plot(number_of_classes_init_10_incr_10, improvements['gamma_0_1']['mean'], 'o--', label='g = 0.1', color="#9467bd")
    plt.plot(number_of_classes_init_10_incr_10, improvements['gamma_0_2']['mean'], 'o-', label='g = 0.2', color="#8c564b")
    plt.plot(number_of_classes_init_10_incr_10, improvements['gamma_0_5']['mean'], 'o-', label='g = 0.5', color="#e377c2")
    plt.plot(number_of_classes_init_10_incr_10, improvements['gamma_1']['mean'], 'o-', label='g = 1', color="#7f7f7f")
    plt.xlabel('Number of Classes')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Lwf Model mean Accuracies')

    plt.tight_layout()
    plt.savefig("resources/lwf_accuracy_improvements_from_ce.png")       
    plt.close()

if acc_impr_with_std:
    # Plot improvement curves with std
    plt.subplots(figsize=(15, 12))
    plt.errorbar(number_of_classes_init_10_incr_10, improvements["gamma_0_01"]["mean"], yerr=improvements["gamma_0_01"]["std"], fmt='o--', capsize=9, label='g = 0.01', color="#2ca02c")
    plt.errorbar(number_of_classes_init_10_incr_10, improvements["gamma_0_02"]["mean"], yerr=improvements["gamma_0_02"]["std"], fmt='o--', capsize=8, label='g = 0.02', color="#d62728")
    plt.errorbar(number_of_classes_init_10_incr_10, improvements["gamma_0_05"]["mean"], yerr=improvements["gamma_0_05"]["std"], fmt='o--', capsize=7, label='g = 0.05', color="#ff7f0e")
    plt.errorbar(number_of_classes_init_10_incr_10, improvements["gamma_0_1"]["mean"], yerr=improvements["gamma_0_1"]["std"], fmt='o--', capsize=6, label='g = 0.1', color="#9467bd")
    plt.errorbar(number_of_classes_init_10_incr_10, improvements["gamma_0_2"]["mean"], yerr=improvements["gamma_0_2"]["std"], fmt='o-', capsize=9, label='g = 0.2', color="#8c564b")
    plt.errorbar(number_of_classes_init_10_incr_10, improvements["gamma_0_5"]["mean"], yerr=improvements["gamma_0_5"]["std"], fmt='o-', capsize=8, label='g = 0.5', color="#e377c2")
    plt.errorbar(number_of_classes_init_10_incr_10, improvements["gamma_1"]["mean"], yerr=improvements["gamma_1"]["std"], fmt='o-', capsize=6, label='g = 1', color="#7f7f7f")
    plt.xlabel('Number of Classes')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Lwf Model accuracy improvements from pure ce')

    plt.tight_layout()
    plt.savefig("resources/lwf_accuracy_improvements_from_ce_with_std.png")       
    plt.close()