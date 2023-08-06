from sklearn.pipeline import Pipeline
from sklearn import set_config
import msci_641_jd_cleaning
import resume_data_extraction
import resume_preprocessing
import encoding_model
import resume_classification

steps = [("Job Description Data Prepration", msci_641_jd_cleaning.main()),
         ("Resume Data Extraction", resume_data_extraction.main()),
         ("Resume Data Pre Preprocessing", resume_preprocessing.main()),
         ("Generating Embeddings and Calculating scores", encoding_model.main()),
         ("Model Classification of the Job title for the resume", resume_classification.main())]

pipe = Pipeline(steps=steps)

#Visualise the pipe
set_config(display="diagram")
print(pipe)