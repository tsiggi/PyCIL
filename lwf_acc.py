import matplotlib.pyplot as plt
import numpy as np

# Data from logs
number_of_classes_init_10_incr_10 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Accuracy of Lwf model with different gamma and seeds for base 10 incr 10 
# - trained for 200 epochs for base task and 150 epochs for incr

gamma_0_seed_0 = [76.3, 61.45, 53.6, 42.92, 38.7, 35.37, 30.34, 26.18, 23.63, 21.29]
gamma_0_seed_1 = [78.2, 64.6, 50.67, 40.7, 35.52, 32.65, 26.17, 26.01, 23.64, 23.37]
gamma_0_seed_2 = [89.4, 73.3, 56.97, 46.15, 37.48, 33.73, 29.43, 27.5, 24.83, 23.47] 
gamma_0_mean = [np.mean([gamma_0_seed_0[i], gamma_0_seed_1[i], gamma_0_seed_2[i]]) for i in range(len(gamma_0_seed_0))]

gamma_0_01_seed_0 = [75.3, 62.4, 54.93, 45.52, 41.38, 36.9, 32.09, 29.0, 25.5, 22.46]
gamma_0_01_seed_1 = [81.2, 68.75, 53.2, 45.48, 39.04, 35.28, 30.5, 28.92, 26.34, 24.97]
gamma_0_01_seed_2 = [90.4, 75.6, 59.77, 47.48, 42.06, 37.93, 31.89, 28.86, 25.9, 25.08]
gamma_0_01_mean = [np.mean([gamma_0_01_seed_0[i], gamma_0_01_seed_1[i], gamma_0_01_seed_2[i]]) for i in range(len(gamma_0_01_seed_0))]

gamma_0_02_seed_0 = [77.6, 62.9, 57.8, 47.2, 43.5, 38.1, 33.09, 29.84, 25.86, 22.13] 
gamma_0_02_seed_1 = [80.7, 68.85, 54.77, 45.88, 39.62, 34.02, 30.24, 29.05, 25.7, 23.55]
gamma_0_02_seed_2 = [89.5, 75.5, 57.5, 47.12, 41.66, 37.7, 31.7, 29.56, 26.97, 25.35]
gamma_0_02_mean = [np.mean([gamma_0_02_seed_0[i], gamma_0_02_seed_1[i], gamma_0_02_seed_2[i]]) for i in range(len(gamma_0_02_seed_0))]

gamma_0_05_seed_0 = [78.2, 63.4, 58.67, 46.92, 42.46, 36.57, 32.53, 30.04, 26.08, 22.14]
gamma_0_05_seed_1 = [82.0, 69.15, 55.23, 46.42, 40.44, 33.32, 30.03, 28.5, 24.88, 22.67]
gamma_0_05_seed_2 = [89.4, 78.25, 59.8, 49.62, 43.02, 37.33, 32.0, 29.28, 26.16, 23.11]
gamma_0_05_mean = [np.mean([gamma_0_05_seed_0[i], gamma_0_05_seed_1[i], gamma_0_05_seed_2[i]]) for i in range(len(gamma_0_05_seed_0))]

gamma_0_1_seed_0 = [74.9, 61.25, 56.2, 46.82, 42.3, 35.8, 31.07, 28.65, 25.09, 21.43]
gamma_0_1_seed_1 = [80.5, 69.15, 53.77, 46.95, 40.4, 33.77, 29.83, 27.44, 24.01, 21.91]
gamma_0_1_seed_2 = [89.2, 77.4, 59.7, 49.32, 42.96, 37.2, 31.43, 27.65, 24.48, 22.36]
gamma_0_1_mean = [np.mean([gamma_0_1_seed_0[i], gamma_0_1_seed_1[i], gamma_0_1_seed_2[i]]) for i in range(len(gamma_0_1_seed_0))]

gamma_0_2_seed_0 = [75.8, 62.25, 56.63, 45.45, 40.94, 34.35, 30.27, 27.25, 23.51, 20.09]
gamma_0_2_seed_1 = [82.2, 70.5, 55.23, 46.82, 40.24, 31.27, 29.63, 27.0, 24.1, 22.29]
gamma_0_2_seed_2 = [90.6, 78.4, 60.1, 48.62, 42.46, 37.12, 29.46, 24.85, 23.21, 21.57]
gamma_0_2_mean = [np.mean([gamma_0_2_seed_0[i], gamma_0_2_seed_1[i], gamma_0_2_seed_2[i]]) for i in range(len(gamma_0_2_seed_0))]

gamma_0_5_seed_0 = [75.9, 62.3, 57.63, 45.22, 41.62, 35.02, 30.26, 26.59, 24.74, 20.65]  
gamma_0_5_seed_1 = [78.1, 68.1, 52.77, 45.5, 39.02, 31.97, 28.3, 25.84, 23.02, 20.28] 
gamma_0_5_seed_2 = [90.0, 78.05, 59.47, 49.0, 40.8, 35.12, 29.44, 25.85, 22.9, 20.79]
gamma_0_5_mean = [np.mean([gamma_0_5_seed_0[i], gamma_0_5_seed_1[i], gamma_0_5_seed_2[i]]) for i in range(len(gamma_0_5_seed_0))]

gamma_1_seed_0 = [76.1, 61.55, 55.77, 44.18, 39.7, 33.55, 29.41, 26.44, 23.12, 19.34]
gamma_1_seed_1 = [81.9, 68.8, 52.83, 45.68, 38.04, 31.57, 28.67, 26.3, 23.68, 21.02]
gamma_1_seed_2 = [90.1, 77.6, 57.9, 47.48, 40.0, 34.38, 27.87, 25.35, 22.79, 20.57]
gamma_1_mean = [np.mean([gamma_1_seed_0[i], gamma_1_seed_1[i], gamma_1_seed_2[i]]) for i in range(len(gamma_1_seed_0))]


# Plot accuracy curves
plt.subplots(figsize=(15, 12))
plt.plot(number_of_classes_init_10_incr_10, gamma_0_mean, 'o--', label='ce', color="#1f77b4")
plt.plot(number_of_classes_init_10_incr_10, gamma_0_01_mean, 'o--', label='g = 0.01', color="#ff7f0e")
plt.plot(number_of_classes_init_10_incr_10, gamma_0_02_mean, 'o--', label='g = 0.02', color="#2ca02c")
plt.plot(number_of_classes_init_10_incr_10, gamma_0_05_mean, 'o--', label='g = 0.05', color="#d62728")
plt.plot(number_of_classes_init_10_incr_10, gamma_0_1_mean, 'o--', label='g = 0.1', color="#9467bd")
plt.plot(number_of_classes_init_10_incr_10, gamma_0_2_mean, 'o-', label='g = 0.2', color="#8c564b")
plt.plot(number_of_classes_init_10_incr_10, gamma_0_5_mean, 'o-', label='g = 0.5', color="#e377c2")
plt.plot(number_of_classes_init_10_incr_10, gamma_1_mean, 'o-', label='g = 1', color="#7f7f7f")
plt.xlabel('Number of Classes')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Lwf Model')

plt.tight_layout()
plt.savefig("resources/lwf_mean_acc.png")       
plt.close()