from sklearn.pipeline import Pipeline
from sklearn import set_config
import msci_641_jd_cleaning
import resume_data_extraction
import resume_preprocessing
import encoding_model

steps = [("Job Description Data Prepration", msci_641_jd_cleaning.main()),
         ("Resume Data Extraction", resume_data_extraction.main()),
         ("Resume Data Pre Preprocessing", resume_preprocessing.main()),
         ("Generating Embeddings and Calculating scores", encoding_model.main())]
# msci_641_jd_cleaning.main()
# resume_data_extraction.main()

pipe = Pipeline(steps=steps)

#Visualise the pipe
set_config(display="diagram")
print(pipe)