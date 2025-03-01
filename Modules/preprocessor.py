import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from Modules import consts

download_path = consts.workspace_dir + '/Data/nltk_data/'
nltk.data.path.append(download_path)


class text_preprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def lemmatize_words(self, text):
        lemmas = [self.lemmatizer.lemmatize(word, pos='v') for word in text]
        return lemmas

    def preprocess(self, txt):

        if not txt:
            return ""

        txt = txt.lower()

        txt = re.sub(consts.URL_PATTERN, '', txt)  # to remove links

        txt = re.sub(r'<.*?>', '', txt)  # to remove html tags </>

        txt = re.sub(r'[^a-z ]', '', txt)  # remove non-alpha character

        # replace multiple spaces with single space
        txt = re.sub(r'\s+', ' ', txt)

        tokenized_txt = word_tokenize(txt)
        words = [word for word in tokenized_txt if (
            (len(word) > 1) and word not in self.stop_words)]
        
        lemmas = self.lemmatize_words(words)

        return lemmas
    
    def clean(self, txt):

        tokenized_txt = self.preprocess(txt)
        txt = ' '.join(tokenized_txt) # remove stop words

        return txt

class job_preprocessor(text_preprocessor):
    def __init__(self):
        super().__init__()

    def clean_job(self, job: dict, keys: list[str]):
        for key in keys:
            job[key] = self.clean(job[key])
        
    def clean_batch(self, data: list[dict], keys: list[str]):
        for item in data:
            self.clean_job(item, keys)
        
        return data