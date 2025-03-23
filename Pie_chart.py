#%% Pie chart example
import matplotlib.pyplot as plt
# Data
labels = ['18-26 Jahre', '27-34 Jahre', '34-40 Jahre']
sizes = [3, 27, 12]  # Absolute values (% needed)
total = sum(sizes)
percentages = [n / total * 100 for n in sizes] # to get percentages

colors = ['yellow', 'blue', 'green'] #['#ff9999','#66b3ff','#99ff99'] #either name them, or specify with #

plt.figure(figsize=(8,10))
plt.pie(percentages, labels=[f'{label} ({p:.1f}%)' for label, p in zip(labels, percentages)],
        autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('Altersgruppenverteilung (N=42)')
plt.axis('equal')  # round shape
plt.show()
# %%
