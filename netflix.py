# ---------------------------------------------------------------
# 📊 The_iScale Movies Data Analysis Project
# ---------------------------------------------------------------

# 🎯 Objective:
# Perform exploratory data analysis (EDA) on a curated movie dataset
# to uncover insights about genre frequency, vote patterns, popularity scores,
# and release trends. Clean and transform the data for meaningful visualizations.

# 📂 Dataset Overview:
# - Source: 'mymoviedb.csv'
# - Original Rows: 9827
# - Final Rows after genre explosion: 25,552
# - Columns used: Release_Date, Title, Popularity, Vote_Count, Vote_Average, Genre

# 🧹 Data Cleaning & Transformation Steps:
# 1. Dropped irrelevant columns: 'Overview', 'Original_Language', 'Poster_Url'
# 2. Converted 'Release_Date' to year only
# 3. Categorized 'Vote_Average' into 4 bins:
#    ['not_popular', 'below_avg', 'average', 'popular']
# 4. Split multi-genre entries and exploded them into separate rows
# 5. Dropped any remaining NaNs

# ❓ Key Questions Explored:
# Q1: What is the most frequent genre in the dataset?
# Q2: Which genres have the highest vote averages?
# Q3: Which movie has the highest popularity score?
# Q4: Which movie has the lowest popularity score?
# Q5: Which year saw the most movie releases?

# 📊 Visualizations Created:
# - Genre distribution bar chart
# - Vote average category count plot
# - Popularity histogram
# - Yearly release distribution

# 🧠 Insights:
# - Drama is the most frequent and most popular genre
# - Popularity and vote count are not always aligned
# - Genre explosion revealed multi-genre patterns
# - Categorizing vote averages made comparisons more intuitive

# ✅ Final Dataset Shape:
# - Rows: 25,552
# - Columns: 6 (Release_Date, Title, Popularity, Vote_Count, Vote_Average, Genre)

# ---------------------------------------------------------------
# End of Project Summary
# ---------------------------------------------------------------

# 1. Exploring Data

# Importing libraries
import numpy as np  # type: ignore
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reading the file
df = pd.read_csv('mymoviedb.csv', lineterminator="\n")
print(df.head())

# View the dataset and datatypes information
print(df.info())

# It suggests that our data has no NaN values. Overview, language, and poster_url are not as useful for us.
# Release_date needs to be casted into date and time.

# Explore genres column
print(df['Genre'].head(10))

# Checking for the duplicates
print("Duplicate rows:", df.duplicated().sum())

# Statistical summary
print(df.describe())

# --- Exploration summary ---
# The dataset contains information about 9,827 movies, each described by 9 features such as title, release date, overview, popularity, vote count, average vote, original language, genre, and poster URL.
# There are no missing values in any column, and the dataset does not contain duplicate rows.
# The 'Genre' column includes multiple genres per movie, and the 'Release_Date' is currently stored as an object (string) rather than a datetime type.
# Numerical columns like 'Popularity', 'Vote_Count', and 'Vote_Average' have been summarized to understand their distributions.
# Overall, the data is clean and ready for further analysis or preprocessing steps.

# 2. Data cleaning

print(df.head())

# Casting Release_Date to datetime
df['Release_Date'] = pd.to_datetime(df['Release_Date'])
print("Release_Date dtype after conversion:", df['Release_Date'].dtypes)

# Extracting year from Release_Date
df['Release_Date'] = df['Release_Date'].dt.year
print("Release_Date dtype after extracting year:", df['Release_Date'].dtypes)

print(df.info())
print(df.head())

# Dropping overview, original_language and Poster_url
cols = ['Overview', 'Original_Language', 'Poster_Url']
df.drop(cols, axis=1, inplace=True)
print("Columns after dropping:", df.columns)
print(df.head())

# Cutting the vote average values and make it in four popular categories: popular, average, below_avg, not_popular

def catigorize_col(df, col, labels):
    """
    Categorizes a certain column based on its quartiles.
    Args:
        df (pd.DataFrame): DataFrame we are processing
        col (str): Column name to be categorized
        labels (list): List of labels from min to max
    Returns:
        pd.DataFrame: DataFrame with the categorized column
    """
    edges = [df[col].describe()['min'],
             df[col].describe()['25%'],
             df[col].describe()['50%'],
             df[col].describe()['75%'],
             df[col].describe()['max']]
    # Remove duplicate edges to avoid errors in pd.cut
    edges = sorted(set(edges))
    # Adjust labels if number of bins is less than labels
    df[col] = pd.cut(df[col], bins=edges, labels=labels[:len(edges)-1], duplicates='drop', include_lowest=True)
    return df

labels = ['not_popular', 'below_avg', 'average', 'popular']
df = catigorize_col(df, 'Vote_Average', labels)
print("Vote_Average unique categories:", df['Vote_Average'].unique())

print(df['Vote_Average'].value_counts())
print(df.head())

# Dropping NaN values (if any)
df.dropna(inplace=True)
print("NaN values after dropping:", df.isna().sum())
print(df.head())

# Split genres into a list and explode the dataframe to have only one genre per row for each movie
df['Genre'] = df['Genre'].str.split(', ')
df = df.explode('Genre').reset_index(drop=True)
print(df.head())

# Casting Genre column into categories
df['Genre'] = df['Genre'].astype('category')
print("Genre dtype after casting:", df['Genre'].dtypes)
print(df.info())

# DATA VISUALIZATION

# Setting seaborn configurations
sns.set_style('whitegrid')

# Q1. What is the most frequent genre in the dataset?
print(df['Genre'].describe())

# Visualizing genre column
sns.catplot(
    y='Genre',
    data=df,
    kind='count',
    order=df['Genre'].value_counts().index,
    color='#4287f5'
)
plt.title('Genre column distribution')
plt.show()

# ANS1. Thus we know that Drama genre is most frequent in the dataset

# Q2. What genres have highest votes?
sns.catplot(
    y='Vote_Average',
    data=df,
    kind='count',
    order=df['Vote_Average'].value_counts().index,
    color='#4287f5'
)
plt.title('VOTES DISTRIBUTION')
plt.show()

# Q3. What movie got the highest popularity? What's its genre?
print(df[df['Popularity'] == df['Popularity'].max()])

# Q4. What movie got the lowest popularity and its genre?
print(df[df['Popularity'] == df['Popularity'].min()])

# Q5. Which year has most filmed movies?
df['Release_Date'].hist()
plt.title('Release_Date column distribution')
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.show()

# --- Final Insights (as comments for GitHub) ---

# Q1: What is the most frequent genre in the dataset?
# Ans. Drama genre is the most frequent genre in our dataset and has appeared more than 14% of the times among 19 other genres.

# Q2: What genres have highest votes?
# Ans. We have 25.5% of our dataset with popular vote (6520 rows). Drama again gets the highest popularity among fans by being having more than 18.5% of movies popularities.

# Q3: What movie got the highest popularity? What's its genre?
# Ans. Spider-Man: No Way Home has the highest popularity rate in our dataset and it has genres of Adventure and Science Fiction.

# Q4: What movie got the lowest popularity? What's its genre?
# Ans. The United States, Thread has the lowest rate in our dataset and it has genres of music, drama, war, sci-fi, and history.

# Q5: Which year has the most filmed movies?
# Ans. 2020 has the highest filming rate in our dataset.