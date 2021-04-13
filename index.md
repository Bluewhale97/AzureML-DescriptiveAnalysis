## Introduction

The traditional statistical analysis in analytics contains two parts, descriptive analysis and inferential analysis. A lot of data science is rooted in statistics, generally we represent measures of central tendency, dispersion, skewness, kurtosis, correlation matrix, specific and interesting features correlation and min, max and outliers analysis in the procedures of descriptive analysis. 

Now we are going to explore some basic statistical techniques using Python in Azure. 

## 1. Frequency and Distribution

When examining a feature, we are interested in its distribution, histogram is a plot that visualize the data and show how frequently each values for a variable occurs.

For example: we use the df_students data set in our previous assignments and plot a histogram for grade of students.

```python
# Get the variable to examine
var_data = df_students['Grade']

# Create a Figure
fig = plt.figure(figsize=(10,4))

# Plot a histogram
plt.hist(var_data)

# Add titles and labels
plt.title('Data Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show the figure
fig.show()
```

![image](https://user-images.githubusercontent.com/71245576/114630542-d6503c80-9c88-11eb-9786-1d5ca1329a22.png)

Intuitively it is really like a Gaussian distribution, but now we should say it is in a symmetric shape without skewness, we need further inference to get to know whether it matches the normality, we will discuss it in another article. 

## 2. Measures of central tendency

Generally, there are three metrics for central tendency analysis: the mean, the median and the mode. The mean of a sample is very sensitive to outliers but the median is not. The mode means that the most commonly occuring value in the sample. As for how to use these three metrics there are some sufficient considerations for different scenarios but here we just discuss how to represent it.

Let's see how to examine min, max, mean, median and mode of grades.

```python
# Get the variable to examine
var = df_students['Grade']

# Get statistics
min_val = var.min()
max_val = var.max()
mean_val = var.mean()
med_val = var.median()
mod_val = var.mode()[0]

print('Minimum:{:.2f}\nMean:{:.2f}\nMedian:{:.2f}\nMode:{:.2f}\nMaximum:{:.2f}\n'.format(min_val,
                                                                                        mean_val,
                                                                                        med_val,
                                                                                        mod_val,
                                                                                        max_val))
```
The result:

![image](https://user-images.githubusercontent.com/71245576/114631044-ddc41580-9c89-11eb-9d8f-4761d5c38db0.png)

As well we can plot the min, max, mean, mode and median all in a single histogram:

```python
# Create a Figure
fig = plt.figure(figsize=(10,4))

# Plot a histogram
plt.hist(var)

# Add lines for the statistics
plt.axvline(x=min_val, color = 'gray', linestyle='dashed', linewidth = 2)
plt.axvline(x=mean_val, color = 'cyan', linestyle='dashed', linewidth = 2)
plt.axvline(x=med_val, color = 'red', linestyle='dashed', linewidth = 2)
plt.axvline(x=mod_val, color = 'yellow', linestyle='dashed', linewidth = 2)
plt.axvline(x=max_val, color = 'gray', linestyle='dashed', linewidth = 2)

# Add titles and labels
plt.title('Data Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Show the figure
fig.show()
```
![image](https://user-images.githubusercontent.com/71245576/114631138-0f3ce100-9c8a-11eb-9431-bc14f369bb80.png)

We could say that for the grade data, the mean, median, and mode all seem to be more or less in the middle of the minimum and maximum, at around 50. Another way to visualize the distribution of a variable is to use a box-and-whisjers plot. Let's create one for the grade data.

```python
# Get the variable to examine
var = df_students['Grade']

# Create a Figure
fig = plt.figure(figsize=(10,4))

# Plot a histogram
plt.boxplot(var)

# Add titles and labels
plt.title('Data Distribution')

# Show the figure
fig.show()
```
The resulting outcome:

![image](https://user-images.githubusercontent.com/71245576/114631322-5c20b780-9c8a-11eb-8f18-76ac60c63e83.png)

There are several methods to combine the frequency plots and the box plots, as I know Violin plot in some packages is really useful to combine them. We can create a function to combine them as well. 

```python
# Create a function that we can re-use
def show_distribution(var_data):
    from matplotlib import pyplot as plt

    # Get statistics
    min_val = var_data.min()
    max_val = var_data.max()
    mean_val = var_data.mean()
    med_val = var_data.median()
    mod_val = var_data.mode()[0]

    print('Minimum:{:.2f}\nMean:{:.2f}\nMedian:{:.2f}\nMode:{:.2f}\nMaximum:{:.2f}\n'.format(min_val,
                                                                                            mean_val,
                                                                                            med_val,
                                                                                            mod_val,
                                                                                            max_val))

    # Create a figure for 2 subplots (2 rows, 1 column)
    fig, ax = plt.subplots(2, 1, figsize = (10,4))

    # Plot the histogram   
    ax[0].hist(var_data)
    ax[0].set_ylabel('Frequency')

    # Add lines for the mean, median, and mode
    ax[0].axvline(x=min_val, color = 'gray', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=mean_val, color = 'cyan', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=med_val, color = 'red', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=mod_val, color = 'yellow', linestyle='dashed', linewidth = 2)
    ax[0].axvline(x=max_val, color = 'gray', linestyle='dashed', linewidth = 2)

    # Plot the boxplot   
    ax[1].boxplot(var_data, vert=False)
    ax[1].set_xlabel('Value')

    # Add a title to the Figure
    fig.suptitle('Data Distribution')

    # Show the figure
    fig.show()

# Get the variable to examine
col = df_students['Grade']
# Call the function
show_distribution(col)
```

The result:

![image](https://user-images.githubusercontent.com/71245576/114631568-d2251e80-9c8a-11eb-802b-501f19cc496c.png)

![image](https://user-images.githubusercontent.com/71245576/114631541-c33e6c00-9c8a-11eb-947a-6bad18eadab7.png)

This function can be reused. In Pandas, dataframe class provides a helpful plot function to show the probability density.
```
def show_density(var_data):
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(10,4))

    # Plot density
    var_data.plot.density()

    # Add titles and labels
    plt.title('Data Density')

    # Show the mean, median, and mode
    plt.axvline(x=var_data.mean(), color = 'cyan', linestyle='dashed', linewidth = 2)
    plt.axvline(x=var_data.median(), color = 'red', linestyle='dashed', linewidth = 2)
    plt.axvline(x=var_data.mode()[0], color = 'yellow', linestyle='dashed', linewidth = 2)

    # Show the figure
    plt.show()

# Get the density of Grade
col = df_students['Grade']
show_density(col)
```

The plot shows:

![image](https://user-images.githubusercontent.com/71245576/114631779-419b0e00-9c8b-11eb-8924-803d431600fe.png)

The density shows the more accurate verification that our data is of a normla distribution within which the mean, mode and median are all at the center and the right tail and left tail are symmetric.

Let's take a look at the distribution of the study hours 

```python
# Get the variable to examine
col = df_students['StudyHours']
# Call the function
show_distribution(col)
```
The result shows that it is totally different from grades distribution. One of differences is that there is a outlier that is far away from the center. 

![image](https://user-images.githubusercontent.com/71245576/114631984-ace4e000-9c8b-11eb-815c-bc809a1069a7.png)

Outliers can be occured for many reasons, specially we shoud distinguish them from leveraged points. There are several methods can be used to detect outliers called anomaly detection. We can discuss it in further articals, but now let's see what will happen if the distribution is without it.

```python
# Get the variable to examine
col = df_students[df_students.StudyHours>1]['StudyHours']
# Call the function
show_distribution(col)
```

We excluded the value 1 which was obviously an outlier for the StudyHours column. See what the new distribution looks like:

![image](https://user-images.githubusercontent.com/71245576/114632285-5b892080-9c8c-11eb-97a3-d0678b00b175.png)

Sometimes the outliers are not clearly showed, we can use the Pandas quantile function to exclude observations below the 0.01th percentile.

```python
q01 = df_students.StudyHours.quantile(0.01)
# Get the variable to examine
col = df_students[df_students.StudyHours>q01]['StudyHours']
# Call the function
show_distribution(col)
```
The result shows:

![image](https://user-images.githubusercontent.com/71245576/114632437-b1f65f00-9c8c-11eb-99fd-b39a580a26d1.png)

Let's look at the density for this new distribution

```python
# Get the density of StudyHours
show_density(col)
```
From the plot, obviously it is rightly skewed with a long tail to the right.

![image](https://user-images.githubusercontent.com/71245576/114632614-0d285180-9c8d-11eb-9e45-beae2b45da42.png)

## 3. Measures of variance

Variance is a critical facet shows the variability of a feature. There are some metrics to measure it: range, variance and standard deviation. They are all widely used in the measure of variance. Let's start.

This is a example to compute the range, variance and sd for Grade and StudyHours.

```python
for col_name in ['Grade','StudyHours']:
    col = df_students[col_name]
    rng = col.max() - col.min()
    var = col.var()
    std = col.std()
    print('\n{}:\n - Range: {:.2f}\n - Variance: {:.2f}\n - Std.Dev: {:.2f}'.format(col_name, rng, var, std))
```

The resulting outcome:

![image](https://user-images.githubusercontent.com/71245576/114633048-dbfc5100-9c8d-11eb-830b-b137414229f4.png)

We should say that the higher the sd, the more variance there is, in another words, the data is more spread out.

Actually, when we think the data is fit with a normal distribution, there are more insights that can be digged out after scaling and normalization. 

Let's see the relationshiop between standard deviation and the data in the normal distribution.

```python
import scipy.stats as stats

# Get the Grade column
col = df_students['Grade']

# get the density
density = stats.gaussian_kde(col)

# Plot the density
col.plot.density()

# Get the mean and standard deviation
s = col.std()
m = col.mean()

# Annotate 1 stdev
x1 = [m-s, m+s]
y1 = density(x1)
plt.plot(x1,y1, color='magenta')
plt.annotate('1 std (68.26%)', (x1[1],y1[1]))

# Annotate 2 stdevs
x2 = [m-(s*2), m+(s*2)]
y2 = density(x2)
plt.plot(x2,y2, color='green')
plt.annotate('2 std (95.45%)', (x2[1],y2[1]))

# Annotate 3 stdevs
x3 = [m-(s*3), m+(s*3)]
y3 = density(x3)
plt.plot(x3,y3, color='orange')
plt.annotate('3 std (99.73%)', (x3[1],y3[1]))

# Show the location of the mean
plt.axvline(col.mean(), color='cyan', linestyle='dashed', linewidth=1)

plt.axis('off')

plt.show()
```

The plot shows that the percentage of data within 1, 2, and 3 standard deviations of the mean (plus or minus).

![image](https://user-images.githubusercontent.com/71245576/114633444-7ceb0c00-9c8e-11eb-8241-992988ae18fc.png)

In any normal distribution:

Approximately 68.26% of values fall within one standard deviation from the mean.
Approximately 95.45% of values fall within two standard deviations from the mean.
Approximately 99.73% of values fall within three standard deviations from the mean.

So, since we know that the mean grade is 49.18, the standard deviation is 21.74, and distribution of grades is approximately normal; we can calculate that 68.26% of students should achieve a grade between 27.44 and 70.92.

For descriptive analysis, there is a built in method called describe for quickly knowing main statistics for all numeric columns.

```python
df_students.describe()
```

The result showed below:

![image](https://user-images.githubusercontent.com/71245576/114633691-0569ac80-9c8f-11eb-9950-e3631690dab3.png)

## 4. Data comparison

This is a process to identify any apparent relationships between variables, we can compare numeric and categorical variables.

Before data comparison, remove rows containing some outliers.

```python
df_sample = df_students[df_students['StudyHours']>1]
df_sample
```

There are two numeric variables and two categocial variabes in the data set, StudentHours and Grade versus Name and Pass.

Let's compare StudyHours and Pass to see if there is any relationship between the number of studying hours and a pass grade.

```python
df_sample.boxplot(column='StudyHours', by='Pass', figsize=(8,5))
```

The result shows an apparent relationship that students who passed the course would spend more hours than students who failed.

![image](https://user-images.githubusercontent.com/71245576/114634111-df90d780-9c8f-11eb-9fe5-3e1dbd521efa.png)

Let's compare two numeric variables by creating a bar chart for grade and study hours

```python
# Create a bar plot of name vs grade and study hours
df_sample.plot(x='Name', y=['Grade','StudyHours'], kind='bar', figsize=(8,5))
```
Because we did not scale the data so that it is not obviously showed that the relationship between grades and study hours.

![image](https://user-images.githubusercontent.com/71245576/114634348-529a4e00-9c90-11eb-9b50-8438ffd7667d.png)

We can normalize the data by using some linrary, i.e., Scikit-Learn which provides a scaler.

```python
from sklearn.preprocessing import MinMaxScaler

# Get a scaler object
scaler = MinMaxScaler()

# Create a new dataframe for the scaled values
df_normalized = df_sample[['Name', 'Grade', 'StudyHours']].copy()

# Normalize the numeric columns
df_normalized[['Grade','StudyHours']] = scaler.fit_transform(df_normalized[['Grade','StudyHours']])

# Plot the normalized values
df_normalized.plot(x='Name', y=['Grade','StudyHours'], kind='bar', figsize=(8,5))
```

![image](https://user-images.githubusercontent.com/71245576/114634419-7b224800-9c90-11eb-9f87-cd95b321e3ba.png)

Now it is easy to say that it seems like students with higher grades tend to have studied more. Why not get a correlation for better understanding of their relationship?

```python
df_normalized.Grade.corr(df_normalized.StudyHours)
```
Woof, the correlation coefficient is about 0.9118, vert strong. Please notice that the correlation is not causation, however.

Now, create a scatter plot for visualizing the correlation:

```python
# Create a scatter plot
df_sample.plot.scatter(title='Study Time vs Grade', x='StudyHours', y='Grade')
```

![image](https://user-images.githubusercontent.com/71245576/114634739-3814a480-9c91-11eb-9e04-d8363f040f3b.png)

Intuitively, I do not think it would fit well as a linear regression but the tutorial wanted me to perform least square regression here. By the way, we can represent linear regression using SciPy in Python.

```python
from scipy import stats

#
df_regression = df_sample[['Grade', 'StudyHours']].copy()

# Get the regression slope and intercept
m, b, r, p, se = stats.linregress(df_regression['StudyHours'], df_regression['Grade'])
print('slope: {:.4f}\ny-intercept: {:.4f}'.format(m,b))
print('so...\n f(x) = {:.4f}x + {:.4f}'.format(m,b))

# Use the function (mx + b) to calculate f(x) for each x (StudyHours) value
df_regression['fx'] = (m * df_regression['StudyHours']) + b

# Calculate the error between f(x) and the actual y (Grade) value
df_regression['error'] = df_regression['fx'] - df_regression['Grade']

# Create a scatter plot of Grade vs StudyHours
df_regression.plot.scatter(x='StudyHours', y='Grade')

# Plot the regression line
plt.plot(df_regression['StudyHours'],df_regression['fx'], color='cyan')

# Display the plot
plt.show()
```

Take a look on the resulting plot:

![image](https://user-images.githubusercontent.com/71245576/114634912-a3f70d00-9c91-11eb-91ee-26c93fad18d9.png)

We also can perform the functionality for how close the predicted values to the actual grades.

```python
# Show the original x,y values, the f(x) value, and the error
df_regression[['StudyHours', 'Grade', 'fx', 'error']]
```

Some of them are dramatically different but some of them are truly close.

![image](https://user-images.githubusercontent.com/71245576/114635080-eddff300-9c91-11eb-8598-cb177ae584ba.png)

Now we can use it to perform some basic prediction.

```python
# Define a function based on our regression coefficients
def f(x):
    m = 6.3134
    b = -17.9164
    return m*x + b

study_time = 14

# Get f(x) for study time
prediction = f(study_time)

# Grade can't be less than 0 or more than 100
expected_grade = max(0,min(100,prediction))

#Print the estimated grade
print ('Studying for {} hours per week may result in a grade of {:.0f}'.format(study_time, expected_grade))
```
The outcome shows me that when a student studies for 14 hours per week may result in a grade of 70.


## Reference:

Explore and analyze data with Python, retrieved from https://docs.microsoft.com/en-us/learn/modules/explore-analyze-data-with-python/


