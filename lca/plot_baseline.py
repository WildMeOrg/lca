import matplotlib.pyplot as plt
import json

# Load results from a JSON file
file_path = '/ekaterina/work/src/lca/lca/tmp/zebra_baseline_prob_098.json'

# Load the JSON data
with open(file_path, 'r') as f:
    results = json.load(f)

results = sorted(results, key=lambda x: x["num human"])

# Extract data
num_human = [res["num human"] for res in results]
precision = [res["precision"] for res in results]
recall = [res["recall"] for res in results]
error_rate = [res["error_rate"] for res in results]
topk = [res["score_threshold"] for res in results]

# Create plot
fig, ax = plt.subplots(3, 1, figsize=(8, 12))


# Plot precision vs num human
ax[0].scatter(num_human, precision, label="Precision", color='b')
ax[0].plot(num_human, precision, color='b')
ax[0].set_xlabel('Num Human')
ax[0].set_ylabel('Precision')
ax[0].set_title('Precision vs Num Human')
# Annotate only every 2nd point to reduce rendering time
for i, txt in enumerate(topk):
    ax[0].annotate(f'{txt}', (num_human[i], precision[i]), textcoords="offset points", xytext=(0,5), ha='center')

# Plot recall vs num human
ax[1].scatter(num_human, recall, label="Recall", color='g')
ax[1].plot(num_human, recall, color='g')
ax[1].set_xlabel('Num Human')
ax[1].set_ylabel('Recall')
ax[1].set_title('Recall vs Num Human')
for i, txt in enumerate(topk):
    ax[1].annotate(f'{txt}', (num_human[i], recall[i]), textcoords="offset points", xytext=(0,5), ha='center')

# Plot error rate vs num human
ax[2].scatter(num_human, error_rate, label="Error Rate", color='r')
ax[2].plot(num_human, error_rate, color='r')
ax[2].set_xlabel('Num Human')
ax[2].set_ylabel('Error Rate')
ax[2].set_title('Error Rate vs Num Human')
for i, txt in enumerate(topk):
    ax[2].annotate(f'{txt}', (num_human[i], error_rate[i]), textcoords="offset points", xytext=(0,5), ha='center')

plt.tight_layout()
plt.show()
