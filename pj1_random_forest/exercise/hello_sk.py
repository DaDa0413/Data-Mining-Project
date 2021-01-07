import sklearn
import seaborn as sns

# Import data set
iris = sns.load_dataset('iris')
print(iris.head())

# Plot
sns.set()
sns.pairplot(iris, hue='species', height=3)