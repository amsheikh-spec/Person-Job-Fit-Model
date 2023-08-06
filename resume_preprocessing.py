import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import seaborn as sns
from tabulate import tabulate
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("data.csv",encoding='latin1')
    df = df[['resume', 'label']]
    title_count = df["label"].value_counts()
    values_to_remove = title_count[title_count < 20].index

    # Filter the DataFrame to remove rows with values less than 20
    resume_filtered = df[~df['label'].isin(values_to_remove)]
    resume_filtered['label'].value_counts()

    # Download stopwords
    nltk.download('stopwords')
    nltk.download('punkt')

    # Convert the resume to lowercase
    resume_filtered['resume'] = resume_filtered['resume'].str.lower()

    # Remove punctuation
    resume_filtered['resume'] = resume_filtered['resume'].str.translate(str.maketrans('', '', string.punctuation))

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    resume_filtered['resume'] = resume_filtered['resume'].apply(lambda x: ' '.join(word for word in word_tokenize(x) if word not in stop_words))

    # Convert the labels to lowercase
    resume_filtered['label'] = resume_filtered['label'].str.lower()

    # Replace underscores with spaces and \n in a label column
    resume_filtered['label'] = resume_filtered['label'].str.replace("\n", "").str.replace("_", " ")

    specific_job_titles = ['software developer', 'systems administrator', 'project manager', 'web developer', 'database administrator', 'java developer', 'network administrator']
    resume_filtered = resume_filtered[resume_filtered['label'].isin(specific_job_titles)]
    resume_filtered["label"].value_counts()
    
    #remove duplicate records
    resume_count = len(resume_filtered)
    print(resume_count)
    
    duplicate_count = resume_filtered.duplicated('resume').sum()
    print("Duplicate records :", duplicate_count)
    
    resume_filtered = resume_filtered.drop_duplicates()
    print("Total resumes available are: ", len(resume_filtered))
    
    df['resume_length'] = df['resume'].str.len()
    max_length = df['resume_length'].max()
    print("Maximum Resume Length:", max_length)
    
    df['resume_word_count'] = df['resume'].apply(lambda x: len(x.split()))
    max_word_count = df['resume_word_count'].max()
    print("Maximum Resume Word Count:", max_word_count)
    
    resume_word_count_stats = df['resume_word_count'].describe()

    resume_word_count_stats = resume_word_count_stats.round(2)

    data = [
        {'Statistic': stat, 'Value': resume_word_count_stats[stat]}
        for stat in resume_word_count_stats.index
    ]

    table_format = 'grid'  # We can use 'plain', 'simple', 'github', 'grid', etc.
    table_headers = ['Statistic', 'Value']
    table = tabulate(data, tablefmt=table_format, headers='keys')

    print(table)

    # Assuming your resumes are stored in the 'resume' column, you can calculate the word count for each resume
    df['resume_word_count'] = df['resume'].apply(lambda x: len(x.split()))

    # Get the word count frequency using value_counts() and sort the results by index (word count)
    word_count_freq = df['resume_word_count'].value_counts().sort_index()

    # Plot the histogram
    plt.bar(word_count_freq.index, word_count_freq.values)
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.title('Word Count Frequency')
    plt.show()

    print(df.isna().sum())
    
    plt.figure(figsize=(8, 6))
    plot = sns.countplot(data=df, x = 'label', order=df['label'].value_counts().index)
    plt.xticks(rotation=90)
    for p in plot.patches:
        height = p.get_height()
        plot.annotate(f'{int(height)}', xy=(p.get_x() + p.get_width() / 2, height), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()

    # Save resumes as CSV
    resume_filtered.to_csv('preprocessed_resume.csv', index=False)
