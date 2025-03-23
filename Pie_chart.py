#%%
import matplotlib.pyplot as plt


#Pie chart example

# Data
labels = ['18-26 Jahre', '27-34 Jahre', '34-40 Jahre']
sizes = [3, 27, 12]  # Absolute values (% needed)
total = sum(sizes)
percentages = [n / total * 100 for n in sizes] # to get percentages

# Optional: colors
colors = ['yellow', 'blue', 'green'] ['#ff9999','#66b3ff','#99ff99']

# Plot
plt.figure(figsize=(6,6))
plt.pie(percentages, labels=[f'{label} ({p:.1f}%)' for label, p in zip(labels, percentages)],
        autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('Altersgruppenverteilung (N=42)')
plt.axis('equal')  # round shape
plt.show()
# %%
