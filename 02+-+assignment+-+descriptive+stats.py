
# coding: utf-8

# # Homework 1 - Part 1

# In the first part of the homework you will be working with a dataset describing various properties of CPUs using the `pandas` library. You will be asked to characterize the data using the tools of descriptive statistics and basic plotting. 
# 
# The homework is graded on a scale from 0 to 100. The first part of the homework is worth a maximum of 40 points while the second part counts for up to 60 points. For each question we indicate how many points you can get. If the answer is not completely correct but nonetheless on the right track, we may decide to give partial credit.

# #### About the data

# The dataset was collected by typing the information on a stack of trump cards in the CPU wars series:
# 
# ![CPU Wars trump cards](cpu_wars.jpg)
# 
# 

# In[1]:


#get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ### Reading the dataset (5 pts)

# With real datasets you often have to do a little grunt work to get the data into a suitable format for your analysis. This is also the case here. We placed the dataset in a shared spreadsheet on Google Drive, which you can access [here](https://docs.google.com/a/johannsen.com/spreadsheets/d/1KsIu6I_TqW0oXwEMpomqfwmmoAz4_-kp6xE8vXDms0w/edit?pli=1#gid=0). If you visit the link, you will be given the opportunity to download the spreadsheet data to our own file.
# 
# While Pandas can read datafiles in many formats, we recommend you use either tab-separated files (tsv) or comma-separated files (csv) for this homework. Complete the steps below:
# 
# 1. Download the file from the link above.
# 2. Place the file in the same directory as the IPython notebook.
# 3. Display the top lines of the *raw* file using the `head` system command. 
# 4. Read the file into a Pandas `DataFrame`, which is bound to the variable `cpus`.

# In[7]:


# with open('CPU Wars - Sheet1.tsv') as cpufile:
#     firstNlines=cpufile.readlines()[0:5]
# firstNlines
#In case the meaning was not to load file into a dataframe and then print the head of the file.

cpus = pd.read_csv('CPU Wars - Sheet1.tsv', '\t')
cpus.head()


# If you successfully completed the above steps, `cpus` should now refer to a `DataFrame` with the columns *name*, *clock_speed*, *bus_speed*, *year*, *n_transistors*, *data_width*, *process*, *die_size*, and *tdp*.

# In[21]:


print(cpus.describe(), "\n")
print(cpus.info())


# ### Find the outlier (5 pts)

# We have deliberately introduced an error in this dataset. Your job is now to find out where it is and remove it. 
# 
# The error is not subtle and should plainly visible if you plot the data against time. The easiest way to do this is to group the `cpus` `DataFrame` by year, taking the mean value of the group, and use the `.plot` method of the resulting structure.
# 
# Make the plot below. Note that you can supply a `subplots=True` argument to the `.plot` method to obtain separate plots for each column.

# In[34]:


cpus.groupby("year").mean().plot(subplots=True)
print(cpus)


# Once you have identified the offending value using the plots, write code below to replace it the correct value, which is **8**. In order to replace a value in a cell, you can use the command `cpus.at`. It works like this:
# 
# `cpus.at[ROW_NUMBER,COLUMN_NAME] = new_value`

# In[38]:


cpus.at[9, "data_width"] = 8
print(cpus[cpus["year"] == 1974])


# ### Correlation analysis (5 pts)

# Make a table showing *only* all the strong correlations between the variables of the dataset. 
# 
# For the purpose of this exercise, we consider a correlation strong whenever *the absolute value* of the correlation coefficient is above 0.9.
# 
# The correlation coefficient should be calculated using **Spearman's method**.
# 
# Hint: Use the `.corr()` method of the `DataFrame`. Check in the documentation (or with the help() command) how you can specify the correlation method.

# In[50]:


cpus_corr = cpus.corr(method="spearman")
np.abs(cpus_corr) > 0.9


# ### Correlation analysis II (5 pts)

# What pairs of variables are strongly negatively correlated? Print out a list of all such pairs. 
# 
# Recall correlation is symmetric, so
# 
# $$\text{corr}(a, b) = \text{corr}(b, a)$$
# 
# Your code should therefore only output each pair of variables once.

# In[80]:


cpus_ordered = cpus_corr.unstack().sort_values()
mask = cpus_ordered.values <= -0.9
cpus_ordered_filtered = pd.Series(cpus_ordered.values[mask], cpus_ordered.index[mask])
cpus_ordered_filtered[0::2]


# ### Exploring Moore's law (20 pts)

# Moore's law refers to a prediction made by Gordon E. More, a co-founder of the Intel Corporation, in 1965. The "law" states that the number of transistors which can be packed in an integrated circuit doubles every second year. Remarkably, the prediction has been empirically verified since 1965 and continues to hold today. 
# 
# In this task we ask you to verify Moore's law using the CPU Wars dataset. In particular you will be recreating the plot below, which is taken from the Wikipedia page on [Moore's law](http://en.wikipedia.org/wiki/Moore's_law).
# 
# ![Moore's law](Moores law Wikipedia.svg)
# 
# The Wikipedia plot has two main components. 
# 
# * A *theoretical line* showing Moore's prediction starting with some value in 1971 and doubling every second year. 
# * The *empirical data points*, which are shown as dots and annotated with the name of the microprocessor.
# 
# We say the law is confirmed when the empirical data points fall at or close to the predicted line.
# 
# Let's begin by constructing the $x$ and $y$ coordinates for the theoretical line, where the x-axis is the year of issue of a microprocessor, and the y-axis is the predicted number of transistors.

# In[118]:


# Replace the two lines below with the minimum and maximum year found in the CPU Wars dataset
min_year = cpus["year"].min()
max_year = cpus["year"].max()

predicted_year = [year for year in range(min_year, max_year + 2) if year % 2 == 0]


# The `predicted_year` list now has an entry for each year in the range, but Moore's law states that the doubling occurs every *second* year. Based on the `predicted_year` list, create a new list that leaves out every other element and assign it to `predicted_year`.

# In[115]:


#the filtering in the list comprehension already did this. 
#but would be like this -> 
# predicted_year = predicted_year[::2]


# Now create a list of predicted transistor counts, `predicted_n_transistors`. It should have the same number of elements as `predicted_year`. 
# 
# The initial prediction, `predicted_n_transistors[0]`, is an empirical value that you read from the CPU Wars dataset. The next prediction is the double of the first prediction. In general, further predictions may be generated as 
# 
# ```
# predicted_n_transistors[i] = 2 * predicted_n_transistors[i-1]
# ```

# In[126]:


predicted_n_transistors = cpus[(cpus["year"] == 1974)]["n_transistors"].values

for i in range(len(predicted_year)-1):
    predicted_n_transistors = np.append(predicted_n_transistors, 2 * predicted_n_transistors[-1])


# Now we are ready to plot the prediction line using Matplotlib.

# In[134]:


fig, ax = plt.subplots()

ax.plot(predicted_year, predicted_n_transistors, label="Moore's law")
# ax.scatter(predicted_year, predicted_n_transistors)
ax.set_yscale('log')
ax.set_xlabel("Year")
ax.set_ylabel("Number of transistors (millions)")
ax.set_title("Moore's law");


# If it worked, the plot should show a straight line from 1974 to 2010

# The next step is to add the data points from the CPU Wars dataset. These should be plotted as unconnected points, which can be accomplished in Matplotlib by using the `.scatter()` command. To see how it works, try substituting `ax.plot` by `ax.scatter` in the cell above.
# 
# The final ingredient in replicating the Wikipedia plot is the command for putting text next a data point, which is called `.annotate`. The basic form is to specify the text you want to place in the plot together with the $(x, y)$ coordinates of its location:
# 
# ```
# ax.annotate("Name of data point", xy=(1985, 11))
# ```
# 
# Place the complete code for making the plot, including the predicted line, empirical data points, and annotations, in the cell below:

# In[147]:


fig, ax = plt.subplots()

ax.plot(predicted_year, predicted_n_transistors, label="Moore's law")
ax.scatter(cpus["year"], cpus["n_transistors"])
# for x,y in cpus[["year", "n_transistors"]].sort_values(["year", "n_transistors"], ascending=[True, False]):
#     print(x,y)
for label, x, y in zip(cpus['name'], cpus['year'], cpus['n_transistors']):
    ax.annotate(label, xy = (x + 0.008, y - 0.003),fontsize = 15)
ax.set_yscale('log')
ax.set_xlabel("Year")
ax.set_ylabel("Number of transistors (millions)")
ax.set_title("Moore's law");

