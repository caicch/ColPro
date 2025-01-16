import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Provided data
tasks = np.arange(1, 9)
accuracy_Colpro = [25.93, 57.40, 46.22, 54.56, 54.42, 59.10, 56.66, 55.14]
accuracy_L2P = [28.48, 56.22, 47.86, 49.24, 56.60, 57.00, 53.68, 52.26]
accuracy_DualPrompt = [27.78, 56.93, 47.75, 52.26, 57.30, 56.67, 53.87, 53.97]
accuracy_ProPrompt = [29.63, 54.50, 48.75, 53.16, 55.50, 58.31, 55.56, 53.95]
accuracy_DAM = [31.48, 55.02, 46.04, 52.86, 54.87, 58.38, 56.35, 53.88]
accuracy_LAE = [29.44, 54.50, 48.06, 53.03, 54.99, 58.67, 55.35, 53.75]

data = pd.DataFrame({
    'Number of Learned Tasks': tasks,
    'Ours': accuracy_Colpro,
    'L2P+': accuracy_L2P,
    'DualPrompt+': accuracy_DualPrompt,
    'ProPrompt': accuracy_ProPrompt,
    'DAM': accuracy_DAM,
    'LAE+': accuracy_LAE
})

# Melt the dataframe for seaborn compatibility
melted_data = data.melt('Number of Learned Tasks', var_name='Method', value_name='Task Accuracy')

# Plotting
plt.figure(figsize=(8, 4))
sns.lineplot(data=melted_data, x='Number of Learned Tasks', y='Task Accuracy', hue='Method', ci=None)
plt.title('Task Accuracy for Different Methods')
plt.ylim(20, 60)
plt.xlabel('Number of Learned Tasks')
plt.ylabel('Task Accuracy')
plt.legend(title='Method')
plt.grid(True)  # Add grid lines

plt.show()
