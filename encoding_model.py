import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np

def model_similarity(model, resumes, jd, designations, k):
  # 'all-MiniLM-L6-v2'
  print(f"Loading pre-trained {model} sentence encoding model")
  model = SentenceTransformer(model)
  print("Embedding resume dataset")
  resume_embeddings = get_embeddings(resumes, model)
  print("Embedding Job Description Dataset")
  jd_embeddings = get_embeddings(jd, model)

  jd = jd_embeddings[0]

  print("Performing cosine similarity")
  #compute cosine similarity of one jd with all the resumes
  cosine_scores = similarity_score(resume_embeddings, jd)

  print("Getting the top K resumes")
  matching_resumes = top_k_resumes("web developer", cosine_scores, resumes, designations, k)
  df = pd.DataFrame(matching_resumes, columns=['Resume', 'Score'])
  print(df)
  return df

def get_embeddings(resumes, model):
  resume_chunk_1 = []
  resume_chunk_2 = []
  resume_chunk_3 = []
  resume_chunk_4 = []
  resume_chunk_5 = []

  #Lets consider the first 1000 tokens with an overlap of 50 tokens we perform split
  for i in range(len(resumes)):
    chunk_1 = []
    chunk_2 = []
    chunk_3 = []
    chunk_4 = []
    chunk_5 = []

    if resumes[i] == "" or resumes[i] is None:
      chunk_1 = [""]
      chunk_2 = [""]
      chunk_3 = [""]
      chunk_4 = [""]
      chunk_5 = [""]

    else: 
      chunks = resumes[i].split()
      chunks_length = len(chunks)

      if(chunks_length > 1050):
        #select the middle of the text works better than truncation 
        mid = int(chunks_length/2)
        chunks = chunks[(mid-500) : (mid+500)]

      if(chunks_length < 250):
        chunk_1 = chunks
      elif(chunks_length > 250 and chunks_length < 450):
        chunk_1 = chunks[:250]
        chunk_2 = chunks[200:]
      elif(chunks_length > 450 and chunks_length < 650):
        chunk_1 = chunks[:250]
        chunk_2 = chunks[200:450]
        chunk_3 = chunks[400:]
      elif(chunks_length > 650 and chunks_length < 850):
        chunk_1 = chunks[:250]
        chunk_2 = chunks[200:450]
        chunk_3 = chunks[400:650]
        chunk_4 = chunks[600:]

      elif(chunks_length > 850 and chunks_length < 1050):
        chunk_1 = chunks[:250]
        chunk_2 = chunks[200:450]
        chunk_3 = chunks[400:650]
        chunk_4 = chunks[600:850]
        chunk_5 = chunks[800:]

      elif(chunks_length > 1050):
        chunk_1 = chunks[:250]
        chunk_2 = chunks[200:450]
        chunk_3 = chunks[400:650]
        chunk_4 = chunks[600:850]
        chunk_5 = chunks[800:1050]
    resume_chunk_1.append(' '.join(chunk_1))
    resume_chunk_2.append(' '.join(chunk_2))
    resume_chunk_3.append(' '.join(chunk_3))
    resume_chunk_4.append(' '.join(chunk_4))
    resume_chunk_5.append(' '.join(chunk_5))


  embed_1 = model.encode(resume_chunk_1)
  embed_2 = model.encode(resume_chunk_2)
  embed_3 = model.encode(resume_chunk_3)
  embed_4 = model.encode(resume_chunk_4)
  embed_5 = model.encode(resume_chunk_5)

  embeddings = [[0]] * len(embed_1)
  #combine Embeddings
  for i in range(len(embed_1)):
    embeddings[i] = []
    embeddings[i].extend(embed_1[i])
    if(i < len(embed_2)):
      embeddings[i].extend(embed_2[i])
    if(i < len(embed_3)):
      embeddings[i].extend(embed_3[i])
    if(i < len(embed_4)):
      embeddings[i].extend(embed_4[i])
    if(i < len(embed_5)):
      embeddings[i].extend(embed_5[i])

  return embeddings

def similarity_score(resumes_embeddings, jd_embedding):
  return util.cos_sim(resumes_embeddings, jd_embedding)

def top_k_resumes(label, cosine_scores, resumes, designations, k):
  values_array = np.array(cosine_scores).reshape(len(resumes))

  # Number of top values to find
  n = 500

  # Get the indices and values of the top k values
  top_k_indices_values = sorted(enumerate(values_array), key=lambda x: x[1], reverse=True)[:n]
  top_k = []
  # Print the indices and values
  for index, value in top_k_indices_values:
      if len(top_k) == k:
        break
      if designations[index] == label and (resumes[index], cosine_scores[index]) not in top_k:
        top_k.append((resumes[index], str(cosine_scores[index] * 100)))
      # print(f"resume: {designations[index]}, Value: {str((value * 100))}")

  return top_k


def main():
  data_resumes = pd.read_csv('/content/drive/MyDrive/MSCI_641/preprocessed_resume.csv')
  resumes = data_resumes["resume"]  # List of resumes
  designations = data_resumes["label"]  # List of corresponding designations
  resumes = list(resumes)
  data_jd = pd.read_csv('preprocessed_jd.csv')
  jd = data_jd["job_description"]
  jd = list(jd)

  sum = 0
  for i in range(len(resumes)):
    tokens = resumes[i].split()
    sum += len(tokens)
  #average token length
  print("Average resume token length is: ", sum/len(resumes))

  allMiniLML6v2 = model_similarity('all-MiniLM-L6-v2', resumes, jd, designations, 5)
  allMiniLML12v2 = model_similarity('all-MiniLM-L12-v2', resumes, jd, designations, 5)
  paraphrasealbertsmallv2 = model_similarity('paraphrase-albert-small-v2', resumes, jd, designations, 5)
  
  skills_data = data_resumes["skills"]
  print("Similarity score of skills with jd")
  model_similarity('all-MiniLM-L6-v2', skills_data, jd, designations, 5)
  model_similarity('all-MiniLM-L12-v2', skills_data, jd, designations, 5)
  model_similarity('paraphrase-albert-small-v2', skills_data, jd, designations, 5)
  