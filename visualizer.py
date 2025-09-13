import matplotlib.pyplot as plt


# Data from logs 
number_of_classes_init_10_incr_10 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
number_of_classes_init_20_incr_20 = [20, 40, 60, 80, 100]
number_of_classes_init_50_incr_10 = [50, 60, 70, 80, 90, 100]
# finetuned model
accuracy_init_10_incr_10 = [91.0, 40.9, 30.97, 21.98, 19.1, 15.2, 13.31, 11.16, 10.52, 8.91]
top1_init_20_incr_20 = [82.3, 42.55, 28.68, 21.26, 17.08]
top1_init_50_incr_10 = [76.88, 14.8, 13.41, 11.42, 10.67, 8.81]
# top5_accuracy_init_10_incr_10 = [99.5, 74.85, 59.93, 50.98, 43.38, 35.8, 29.11, 28.54, 24.52, 19.87]
# top5_init_20_incr_20 = [97.0, 58.3, 47.33, 37.51, 28.45]
# top5_init_50_incr_10 = [94.62, 38.75, 27.36, 27.04, 23.93, 19.43]
# replay model
rep_top1_init_10_incr_10 = [89.8, 78.1, 70.3, 62.55, 58.86, 54.12, 51.83, 45.65, 43.27, 41.8]
rep_top1_init_20_incr_20 = [81.25, 70.45, 59.37, 50.36, 44.73]
rep_top1_init_50_incr_10 = [77.34, 54.08, 51.93, 46.59, 44.83, 42.87]
# rep_top5_init_10_incr_10 = [99.3, 96.2, 92.23, 88.72, 85.54, 82.43, 80.31, 77.62, 75.04, 73.09]
# rep_top5_init_20_incr_20 = [96.95, 93.32, 87.05, 81.6, 77.1]
# rep_top5_init_50_incr_10 = [94.6, 83.88, 81.94, 78.39, 76.16, 73.79]

# onother model
lwf_10_10 = [91.4, 67.8, 57.2, 43.48, 39.98, 33.1, 31.24, 25.35, 25.0, 23.34]
lwf_20_20 = [82.15, 61.18, 47.83, 40.24, 35.79]
lwf_50_10 = [75.86, 47.7, 36.47, 27.34, 24.5, 23.09]

# Plot accuracy curves
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
# axes[0].plot(number_of_classes_init_10_incr_10, top5_accuracy_init_10_incr_10, 'o--', label='Finetuned top5 accuracy', color="#870C0C")
# axes[0].plot(number_of_classes_init_10_incr_10, rep_top5_init_10_incr_10, 'o--', label='Replay top5 accuracy', color="#02198D")
axes[0].plot(number_of_classes_init_10_incr_10, accuracy_init_10_incr_10, 'o-', label='Finetuned top1 accuracy', color="#CE1616" )
axes[0].plot(number_of_classes_init_10_incr_10, rep_top1_init_10_incr_10, 'o-', label='Replay top1 accuracy', color="#2341D7" )
axes[0].plot(number_of_classes_init_10_incr_10, lwf_10_10, 'o-', label='Lwf top1 accuracy', color="#1f77b4" )
axes[0].set_xlabel('Number of Classes')
axes[0].set_ylabel('Accuracy (%)')
axes[0].legend()
axes[0].set_title('10 tasks of 10 classes each')


# axes[1].plot(number_of_classes_init_20_incr_20, top5_init_20_incr_20, 'o--', label='Finetuned top5 accuracy', color="#870C0C")
# axes[1].plot(number_of_classes_init_20_incr_20, rep_top5_init_20_incr_20, 'o--', label='Replay top5 accuracy', color="#02198D")
axes[1].plot(number_of_classes_init_20_incr_20, top1_init_20_incr_20, 'o-', label='Finetuned top1 accuracy', color="#CE1616")
axes[1].plot(number_of_classes_init_20_incr_20, rep_top1_init_20_incr_20, 'o-', label='Replay top1 accuracy', color="#2341D7" )
axes[1].plot(number_of_classes_init_20_incr_20, lwf_20_20, 'o-', label='Lwf top1 accuracy', color="#1f77b4" )
axes[1].set_xlabel('Number of Classes')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()
axes[1].set_title('5 tasks of 20 classes each')

# axes[2].plot(number_of_classes_init_50_incr_10, top5_init_50_incr_10, 'o--', label='Finetuned top5 accuracy', color="#870C0C")
# axes[2].plot(number_of_classes_init_50_incr_10, rep_top5_init_50_incr_10, 'o--', label='Replay top5 accuracy', color="#02198D")
axes[2].plot(number_of_classes_init_50_incr_10, top1_init_50_incr_10, 'o-', label='Finetuned top1 accuracy', color="#CE1616")
axes[2].plot(number_of_classes_init_50_incr_10, rep_top1_init_50_incr_10, 'o-', label='Replay top1 accuracy', color="#2341D7" )
axes[2].plot(number_of_classes_init_50_incr_10, lwf_50_10, 'o-', label='Lwf top1 accuracy', color="#1f77b4" )
axes[2].set_xlabel('Number of Classes')
axes[2].set_ylabel('Accuracy (%)')
axes[2].legend()
axes[2].set_title('5 tasks of 10 classes each (initial 50 classes)')

plt.tight_layout()
# plt.show()
plt.savefig("resources/cl_models.png")       # PNG image
plt.close()