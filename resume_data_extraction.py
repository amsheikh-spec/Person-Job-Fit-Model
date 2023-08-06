import zipfile
import pandas as pd
from bs4 import BeautifulSoup
import re

def extract_skills(resume_text):
    # Define a regular expression pattern to find the "Skills" section
    skills_pattern = r"\b(?:Skills|Technical Skills|Key Skills)\b"

    # Find the start index of the "Skills" section
    start_match = re.search(skills_pattern, resume_text, flags=re.IGNORECASE)
    if start_match:
        start_index = start_match.start()
    else:
        # The "Skills" section header was not found
        return None

    # Extract the content after the "Skills" section header
    skills_content = resume_text[start_index + len(start_match.group()):]

    # Identify the end of the "Skills" section (e.g., the next section header)
    end_match = re.search(
        r"\b(?:Experience|Education|Projects|Certifications|References|Education|Links)\b",
        skills_content,
        flags=re.IGNORECASE,
    )
    if end_match:
        end_index = end_match.start()
    else:
        # If no next section header is found, assume the end of the resume
        end_index = len(skills_content)

    # Extract the "Skills" section content
    skills_section = skills_content[:end_index].strip()

    # Split the skills content into a list of individual skills
    skills_list = [skill.strip() for skill in skills_section.split("\n") if skill.strip()]

    return ''.join(skills_list)
  
def main():
  data = []
  # Specify the path to the ZIP file
  zip_path = 'data/resumes_corpus.zip'

  # Open the ZIP file in read mode
  with zipfile.ZipFile(zip_path, 'r') as zip_ref:
      # Get a list of all the files and directories inside the ZIP file
      file_list = zip_ref.namelist()
      print("Lenght : ", len(file_list))

      # Iterate over each file in the ZIP file
      for file_name in file_list:
        # if len(data) % 10000 == 0:
        #   print("Progress: ", len(data) * 100/ len(file_list))
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

  df['skills'] = df['resume'].apply(extract_skills)
  # Save DataFrame as CSV
  df.to_csv('processed data/extracted_resumes.csv', index=False)


if __name__ == "__main__":
  main()
  print("Successfully Created Resume data and saved as CSV")
            