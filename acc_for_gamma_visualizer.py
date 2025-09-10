import matplotlib.pyplot as plt


# Data from logs
number_of_classes_init_10_incr_10 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# FINETUNE MODEL WITH DIFFERENT GAMMA VALUES
gamma_0 = [91.0, 40.9, 30.97, 21.98, 19.1, 15.2, 13.31, 11.16, 10.52, 8.91]
gamma_0_01 = [90.3, 40.7, 31.2, 22.0, 18.58, 14.87, 13.1, 11.29, 10.18, 8.71]
gamma_0_02 = [89.9, 41.8, 30.87, 21.98, 18.84, 14.98, 12.96, 11.11, 10.3, 8.69]
gamma_0_05 = [87.7, 38.35, 30.47, 21.78, 18.5, 14.82, 13.11, 10.96, 10.24, 8.36]
gamma_0_1 = [91.1, 40.65, 30.7, 22.02, 18.82, 14.68, 12.96, 11.2, 9.99, 8.6]
gamma_0_2 = [89.1, 39.6, 29.97, 21.78, 18.46, 14.65, 13.04, 11.02, 9.93, 8.53]
gamma_0_5 = [90.1, 40.4, 30.47, 21.85, 18.74, 14.87, 12.97, 11.01, 10.06, 8.62]
gamma_1 = [90.9, 40.65, 30.07, 21.95, 18.78, 14.87, 12.99, 10.95, 10.08, 8.6]

# REPLAY MODEL WITH DIFFERENT GAMMA VALUES
r_gamma_0 = [89.8, 78.1, 70.3, 62.55, 58.86, 54.12, 51.83, 45.65, 43.27, 41.8]
r_gamma_0_01 = [90.1, 77.25, 70.4, 61.3, 57.22, 52.83, 49.53, 44.49, 42.18, 40.82]
r_gamma_0_02 = [90.7, 77.15, 70.3, 61.88, 56.62, 52.98, 50.04, 43.22, 40.51, 39.58]
r_gamma_0_05 = [89.2, 77.45, 70.63, 62.02, 56.04, 50.47, 48.87, 43.18, 41.3, 39.66]
r_gamma_0_1 = [90.2, 76.9, 70.47, 62.3, 57.12, 52.12, 48.56, 43.11, 41.13, 39.38]
r_gamma_0_2 = [89.2, 76.4, 69.33, 62.05, 56.44, 51.33, 48.3, 42.31, 40.08, 37.48]
r_gamma_0_5 = [90.2, 76.8, 70.4, 62.52, 57.22, 51.53, 47.7, 42.7, 38.83, 37.99]
r_gamma_1 = [89.9, 76.35, 70.83, 63.0, 57.28, 50.82, 47.71, 40.52, 39.5, 36.01]


# Plot accuracy curves
fig, axes = plt.subplots(1, 2, figsize=(12, 8))
axes[0].plot(number_of_classes_init_10_incr_10, gamma_0, 'o--', label='ce', color="#1f77b4")
axes[0].plot(number_of_classes_init_10_incr_10, gamma_0_01, 'o--', label='g = 0.01', color="#ff7f0e")
axes[0].plot(number_of_classes_init_10_incr_10, gamma_0_02, 'o--', label='g = 0.02', color="#2ca02c")
axes[0].plot(number_of_classes_init_10_incr_10, gamma_0_05, 'o--', label='g = 0.05', color="#d62728")
axes[0].plot(number_of_classes_init_10_incr_10, gamma_0_1, 'o--', label='g = 0.1', color="#9467bd")
axes[0].plot(number_of_classes_init_10_incr_10, gamma_0_2, 'o-', label='g = 0.2', color="#8c564b")
axes[0].plot(number_of_classes_init_10_incr_10, gamma_0_5, 'o-', label='g = 0.5', color="#e377c2")
axes[0].plot(number_of_classes_init_10_incr_10, gamma_1, 'o-', label='g = 1', color="#7f7f7f")
axes[0].set_xlabel('Number of Classes')
axes[0].set_ylabel('Accuracy (%)')
axes[0].legend()
axes[0].set_title('Finetune Model')

axes[1].plot(number_of_classes_init_10_incr_10, r_gamma_0, 'o--', label='ce', color="#1f77b4")
axes[1].plot(number_of_classes_init_10_incr_10, r_gamma_0_01, 'o--', label='g = 0.01', color="#ff7f0e")
axes[1].plot(number_of_classes_init_10_incr_10, r_gamma_0_02, 'o--', label='g = 0.02', color="#2ca02c")
axes[1].plot(number_of_classes_init_10_incr_10, r_gamma_0_05, 'o--', label='g = 0.05', color="#d62728")
axes[1].plot(number_of_classes_init_10_incr_10, r_gamma_0_1, 'o--', label='g = 0.1', color="#9467bd")
axes[1].plot(number_of_classes_init_10_incr_10, r_gamma_0_2, 'o-', label='g = 0.2', color="#8c564b")
axes[1].plot(number_of_classes_init_10_incr_10, r_gamma_0_5, 'o-', label='g = 0.5', color="#e377c2")
axes[1].plot(number_of_classes_init_10_incr_10, r_gamma_1, 'o-', label='g = 1', color="#7f7f7f")
axes[1].set_xlabel('Number of Classes')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()
axes[1].set_title('Replay Model')

plt.tight_layout()
plt.show()
