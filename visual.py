import matplotlib.pyplot as plt
import numpy as np

# Data from your trials
trials = [
    {
        "label": "Trial 1\n(lr=2.64e-05,\nfreeze=1,\nepochs=4)",
        "BERT_acc": 89.08,
        "RoBERTa_acc": 86.78,
        "Avg_acc": 87.93
    },
    {
        "label": "Trial 2\n(lr=2.12e-05,\nfreeze=6,\nepochs=4)",
        "BERT_acc": 81.03,
        "RoBERTa_acc": 85.06,
        "Avg_acc": 83.05
    },
    {
        "label": "Trial 3\n(lr=2.62e-05,\nfreeze=6,\nepochs=2)",
        "BERT_acc": 67.24,
        "RoBERTa_acc": 81.03,
        "Avg_acc": 74.14
    }
]

labels = [t["label"] for t in trials]
bert_accuracies = [t["BERT_acc"] for t in trials]
roberta_accuracies = [t["RoBERTa_acc"] for t in trials]
avg_accuracies = [t["Avg_acc"] for t in trials]

x = np.arange(len(labels))  # the label locations
width = 0.25                # width of the bars

fig, ax = plt.subplots(figsize=(8, 5))

# Plot bars for each model’s accuracy
rects1 = ax.bar(x - width, bert_accuracies, width, label='BERT', color='tab:blue')
rects2 = ax.bar(x, roberta_accuracies, width, label='RoBERTa', color='tab:orange')
rects3 = ax.bar(x + width, avg_accuracies, width, label='Average', color='tab:green')

# Add text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy by Hyperparameter Settings (BERT, RoBERTa, and Avg)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim([0, 100])  # accuracies from 0% to 100%
ax.legend()

# Optionally, add a function to label the bars with their heights.
def autolabel(rects):
    """Attach a text label above each bar."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  # 4 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.show()