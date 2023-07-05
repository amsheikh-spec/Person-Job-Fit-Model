import zipfile
import pandas as pd
from bs4 import BeautifulSoup

data = []
# Specify the path to the ZIP file
zip_path = '/data/resumes_corpus.zip'

# Open the ZIP file in read mode
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Get a list of all the files and directories inside the ZIP file
    file_list = zip_ref.namelist()
    print("Lenght : ", len(file_list))

    # Iterate over each file in the ZIP file
    for file_name in file_list:
      if len(data) % 1000 == 0:
        print("Progress: ", len(data) * 100/ len(file_list))
      # Read the contents of the file
      if file_name[-3:] != "lab":
        with zip_ref.open(file_name, 'r') as file:
          contents = file.read()
          with zip_ref.open(file_name[:-4] + ".lab", 'r') as file:
            lab = file.readline()
            lab = lab.decode('utf-8')
            cleantext = BeautifulSoup(contents, "lxml").text
            resume = [cleantext, lab]
        data.append(resume)
        
# Convert list to DataFrame
df = pd.DataFrame(data, columns=['resume', 'label'])

# Save DataFrame as CSV
df.to_csv('data.csv', index=False)
            