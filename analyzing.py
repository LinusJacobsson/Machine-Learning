# Pre analyzing of the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import seaborn as sns


#We first load csv w pandas.
# Total length 1039

data = pd.read_csv('train.csv')
print("Head of data")
print(data.head())

print("Descripiton of data")
print(data.describe())

data.replace("Female", True, inplace = True)
data.replace("Male", False, inplace = True)


plt.figure(1)
sns.heatmap(data.corr())




x_data = data.drop(columns = "Lead")
y_data = data["Lead"]
y_data.replace("Female", 1, inplace = True)
y_data.replace("Male", 0, inplace = True)


print(data.corr()["Lead"])

#plt.figure(2)
#plt.bar(data.corr()["Lead"])

#print(data.corr())

# We are interested in the following plots
# Men vs women word count
# Men vs women word count over years?
# Correlation between male speech % and gross income


# Question 1: Do men or women speak more?
men_pct_speech = (data["Number words male"])/(data["Number words female"] + data["Number words male"])  # a value of above 1 signifies male dominance.
print("Men speak on average: " + str(np.mean(men_pct_speech)) + " percent of all movies")
# Men speak on avg 0.65433 pct of movies. Men tend to dominate

# Question 2: Has gender balance changed?

years = data["Year"].values
order = np.argsort(years)
men_pct_speech_year = np.array([np.mean(men_pct_speech[data["Year"] == e]) for e in years])

plt.figure(2)
plt.rc('font', size=12)
#plt.rc('xtick', labelsize=12)
plt.rc('axes', labelsize=14)
#plt.rc('ytick', labelsize=12)
plt.plot(years[order], men_pct_speech_year[order])
plt.title("Average percent words from men in movies for different years")
plt.xlabel("Year")
plt.ylabel("Average words from men [%]")
plt.grid(linestyle='-')

plt.figure(3)
plt.scatter(years[order], men_pct_speech[order])
plt.show()