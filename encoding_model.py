import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np

def top_resumes(label, cosine_scores, resumes, designations):
  values_array = np.array(cosine_scores).reshape(len(cosine_scores))

  # Number of top values to find
  k = 500

  # Get the indices and values of the top k values
  top_k_indices_values = sorted(enumerate(values_array), key=lambda x: x[1], reverse=True)[:k]
  top_5 = []
  # Print the indices and values
  for index, value in top_k_indices_values:
      if len(top_5) == 5:
        break
      if designations[index] == label and (resumes[index], cosine_scores[index]) not in top_5:
        top_5.append((resumes[index], str(cosine_scores[index] * 100)))
      # print(f"resume: {designations[index]}, Value: {str((value * 100))}")

  return top_5


def main():
    data_resumes = pd.read_csv('processed data/preprocessed_resume.csv')
    resumes = data_resumes["resume"]  # List of resumes
    designations = data_resumes["label"]  # List of corresponding designations

    model = SentenceTransformer('all-MiniLM-L6-v2')
    resume_embeddings = model.encode(resumes)

    data_jd = pd.read_csv('preprocessed_jd.csv')
    jd = data_jd["job_description"]  # List of resumes
    software_dev_jd = jd[0]
    project_manager_jd = jd[4]
    web_dev_jd = jd[12]
    java_dev_jd = jd[15]
    database_admin_jd = jd[34]
    network_admin_jd = jd[2]
    system_admin_jd = jd[21]

    software_dev_jd = model.encode(software_dev_jd)
    project_manager_jd = model.encode(project_manager_jd)
    web_dev_jd = model.encode(web_dev_jd)
    java_dev_jd = model.encode(java_dev_jd)
    database_admin_jd = model.encode(database_admin_jd)
    network_admin_jd = model.encode(network_admin_jd)
    system_admin_jd = model.encode(system_admin_jd)

    #Compute cosine-similarities for each sentence with each other sentence
    soft_cosine_scores = util.cos_sim(resume_embeddings, software_dev_jd)
    pm_cosine_scores = util.cos_sim(resume_embeddings, project_manager_jd)
    wd_cosine_scores = util.cos_sim(resume_embeddings, web_dev_jd)
    jvd_cosine_scores = util.cos_sim(resume_embeddings, java_dev_jd)
    da_cosine_scores = util.cos_sim(resume_embeddings, database_admin_jd)
    na_cosine_scores = util.cos_sim(resume_embeddings, network_admin_jd)
    sa_cosine_scores = util.cos_sim(resume_embeddings, system_admin_jd)

    #similarly we can find the top list of resumes for other job postings 
    sd_resumes = top_resumes("software developer", soft_cosine_scores, resumes, designations)
    pm_resumes = top_resumes("project manager", pm_cosine_scores, resumes, designations)
    na_resumes = top_resumes("network administrator", na_cosine_scores, resumes, designations)
    jvd_resumes = top_resumes("java developer", jvd_cosine_scores, resumes, designations)
    wd_resumes = top_resumes("web developer", wd_cosine_scores, resumes, designations)
    da_resumes = top_resumes("database administrator", da_cosine_scores, resumes, designations)
    sa_resumes = top_resumes("systems administrator", sa_cosine_scores, resumes, designations)

    df = pd.DataFrame(wd_resumes, columns=['Resume', 'Score'])
    print(df)

    label_map = {'software developer': 1, 'project manager': 2, 'web developer': 3, 'java developer': 4, 'database administrator': 5, 'network administrator': 6, 'systems administrator': 7}
    data_resumes['job_title_label'] = data_resumes['label'].map(label_map)
    data_jd['job_title_label'] = data_jd['title'].map(label_map)

    data_resumes.head()

