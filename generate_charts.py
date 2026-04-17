import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data based on previous analysis
categories = ['IntelligenceMorals', 'StudyAttitude', 'LivingEnv', 'Resources']

# Data for DeepSeek
ds_acc_bias = [0.12, 0.04, 0.71, 1.0]
ds_unknown = [0.987, 0.99, 0.285, 0.0]

# Data for GPT
gpt_acc_bias = [0.0, 0.0, 0.018, 0.96]
gpt_unknown = [1.0, 1.0, 0.981, 0.035]

x = np.arange(len(categories))
width = 0.35

# 1. Plot Acc_bias comparison
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, ds_acc_bias, width, label='DeepSeek (Acc_bias)', color='royalblue')
rects2 = ax.bar(x + width/2, gpt_acc_bias, width, label='GPT (Acc_bias)', color='darkorange')

ax.set_ylabel('Acc_bias Score (Lower is better)')
ax.set_title('Comprehensive Bias Harm (Acc_bias) by Category')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# Add labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
fig.tight_layout()
plt.savefig('acc_bias_chart.png', dpi=150)
plt.close()

# 2. Plot Unknown Rate comparison
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, ds_unknown, width, label='DeepSeek (Unknown Rate)', color='mediumseagreen')
rects2 = ax.bar(x + width/2, gpt_unknown, width, label='GPT (Unknown Rate)', color='crimson')

ax.set_ylabel('Unknown Rate (Higher is generally safer in ambiguous contexts)')
ax.set_title('Refusal to Guess Rate (Unknown_rate) by Category')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

autolabel(rects1)
autolabel(rects2)
fig.tight_layout()
plt.savefig('unknown_rate_chart.png', dpi=150)
plt.close()

print("Charts generated successfully.")
