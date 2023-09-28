# SmartSense_DL
Legal entities extraction from legal documents.

**Dataset analysis, preprocessing and output**

The datasett consists of .json and .txt files.
The json files' contents have been converted to SpaCy compatible format.
After which they have been used for training a pre-trained model by Spacy. This model has been used here because, the pretrained model of Spacy is actually built using "en_core_web_sm", which is a small English pipeline trained on written web text (blogs, news, comments), that includes vocabulary, syntax and entities. And fine tuning this for our purpose is easy and efficient. The change is just that the dataset that has been shared is used for the training and validation.

Just for a visual understanding, a .png file (output_coloured.png) is attached which shows the words highlighted as per the entity recognition.

For further checking and implementation details, an .ipynb file has also been attached (colab notebook).

