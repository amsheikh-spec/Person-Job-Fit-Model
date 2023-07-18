import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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
    
    # Save resumes as CSV
    resume_filtered.to_csv('preprocessed_resume.csv', index=False)
