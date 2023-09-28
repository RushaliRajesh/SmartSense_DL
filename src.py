import json
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm

'''getting the data ready in Spacy format'''

train_data = []
with open('datasets_dl/datasets/6_3.json', 'r', encoding='utf-8') as f:
    # Iterate through lines in the file
    for line in f:
        # Parse each line as a JSON object
        example = json.loads(line)
        
        # Extract the "text" and "entities" fields
        text = example['text']
        entities = example['entities']
        
        # Create a dictionary in the desired format
        temp_dict = {'text': text, 'entities': entities}
        
        # Append the dictionary to the train_data list
        train_data.append(temp_dict)


nlp = spacy.blank("en")
doc_bin = DocBin()

from spacy.util import filter_spans

for training_example in tqdm(train_data):
    text = training_example['text']
    labels = training_example['entities']
    doc = nlp.make_doc(text)
    ents = []
    for start, end, label in labels:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    filtered_ents = filter_spans(ents)
    doc.ents = filtered_ents
    doc_bin.add(doc)

doc_bin.to_disk("train.spacy")

import subprocess

# Specify the command as a list of strings
command = [
    'python',
    '-m',
    'spacy',
    'init',
    'fill-config',
    'base_config.cfg',
    'config.cfg'
]

# Run the command
subprocess.run(command, shell=True, check=True)

import subprocess

# Specify the command as a list of strings
command = [
    'python',
    '-m',
    'spacy',
    'train',
    'config.cfg',  # Path to your configuration file
    '--output',
    './',  # Output directory
    '--paths.train',
    './train.spacy',  # Path to your training data
    '--paths.dev',
    './train.spacy'  # Path to your development data
]
subprocess.run(command, shell=True, check=True)

import subprocess

# Define the command as a list of arguments
command = ["python", "-m", "spacy", "init", "fill-config", "/content/base_config.cfg", "/content/config.cfg"]
subprocess.run(command, check=True, shell=True)


from spacy import displacy

# Load the pre-trained NER model
nlp_ner = spacy.load("model-best")

# Read the contents of the text file
with open('/content/datasets/legal_train.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Process the text with spaCy
doc = nlp_ner(text)

# Define entity colors
colors = {"Organization": "#F67DE3", "person": "#7DF6D9", "Courts": "#a6e22d"}

# Set rendering options
options = {"colors": colors} 

# Visualize the named entities
displacy.render(doc, style="ent", options=options, jupyter=True)
