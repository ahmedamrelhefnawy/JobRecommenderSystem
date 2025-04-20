import re
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from abc import ABC, abstractmethod

from Modules.model_handlers import model_loader, embed
import Modules.consts as consts

from pydantic import BaseModel
from Models.models import user, job

download_path = consts.workspace_dir + '/Data/nltk_data/'
nltk.data.path.append(download_path)

loader = model_loader()


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

        tokenized_txt = self.preprocess(txt)  # remove stop words
        txt = ' '.join(tokenized_txt)

        return txt


class obj_preprocessor(text_preprocessor):
    """
    Preprocesses objects.
    Objects can be jobs or users.
    """

    def clean_obj(self, obj: user | job, attrs: list[str]):
        """
        Cleans the object.
        """
        for attribute in attrs:
            value = getattr(obj, attribute)
            cleaned_value = self.clean(value)
            setattr(obj, attribute, cleaned_value)

    def clean_batch(self, obj_list: list[user | job], attrs: list[str]):
        """
        Cleans batch of objects.
        """
        for obj in obj_list:
            self.clean_obj(obj, attrs)

        return obj_list


class embedder(ABC):
    def __init__(self,
                 vectorizers: dict = None,
                 encoders: dict = None,
                 preprocessor=None,
                 ):

        self.vectorizers = vectorizers
        self.encoders = encoders
        self.preprocessor = preprocessor

        # Handle missing vectorizers
        if not self.vectorizers:
            self.vectorizers = dict()

        # Handle missing encoders
        if not self.encoders:
            self.encoders = dict()

        if not self.preprocessor:
            self.preprocessor = text_preprocessor()

    @abstractmethod
    def preprocess(self, obj: type[BaseModel]):
        pass
    
    @abstractmethod
    def embed(self, obj: type[BaseModel]):
        pass

    def embed_batch(self, objs: list[type[BaseModel]]):
        return [self.embed(obj) for obj in objs]


class ModelEmbedder(embedder):
    
    """
    Base class for embedding Pydantic models.
    """
    model_class: type[BaseModel] = None  # Will be set by child classes

    def __init__(self,
                 vectorizers: dict = None,
                 encoders: dict = None,
                 preprocessor= None):
        super().__init__(vectorizers, encoders, preprocessor)

        if not self.model_class:
            raise ValueError("model_class must be set by child classes")

        # Load vectorizers and encoders based on configuration
        self._load_models()

        # Check for conflicts
        intersect_keys = set(self.vectorizers.keys()).intersection(self.encoders.keys())
        if intersect_keys:
            raise ValueError(
                f"Can't Assign two models to the same feature\nConflict with features: {intersect_keys}")

    @abstractmethod
    def _load_models(self):
        """Define which models to load for vectorizers and encoders"""
        pass

    def embed(self, obj: BaseModel):
        """
        Embeds a single model instance.
        """
        obj = self.preprocess(obj)
        
        for feature in self.model_class.model_fields.keys():
            value = getattr(obj, feature)

            if feature in self.vectorizers:
                obj_embd = embed(self.vectorizers[feature], [value])
                setattr(obj, feature, obj_embd)

            elif feature in self.encoders:
                obj_embd = embed(self.encoders[feature], value)

                if len(obj_embd) > 1:
                    obj_embd = np.sum(obj_embd, axis=0, keepdims=True)
                
                setattr(obj, feature, obj_embd)
        
        return obj

    def embed_batch(self, objs: list[BaseModel]):
        """
        Embeds batch of objects.
        """
        embeddings = []
        for obj in objs:
            obj_embd = self.embed(obj)
            embeddings.append(obj_embd)

        return embeddings


class job_embedder(ModelEmbedder):
    model_class = job

    def _load_models(self):
        # Load Title Vectorizer
        if not ('title' in self.vectorizers and self.vectorizers['title']):
            self.vectorizers['title'] = loader.load_vectorizer('MiniLM', model_type='sentence_transformer')

        # Load Content Vectorizer
        if not ('content' in self.vectorizers and self.vectorizers['content']):
            self.vectorizers['content'] = loader.load_vectorizer('jobs_tfidf.pkl', model_type='sklearn')

        # Load Encoders
        if not ('work_type' in self.encoders and self.encoders['work_type']):
            self.encoders['work_type'] = loader.load_encoder('work_type_onehot_enc.pkl', model_type='sklearn')
    
    def preprocess(self, obj: job):
        # Clean the title and content
        obj.title = self.preprocessor.clean(obj.title)
        obj.content = self.preprocessor.clean(obj.content)
        
        # Put work type in a list for being compatible with the encoder
        obj.work_type = [[obj.work_type]]
        
        return obj


class user_embedder(ModelEmbedder):
    model_class = user

    def _load_models(self):
        # Load Title Vectorizer
        if not ('title' in self.vectorizers and self.vectorizers['title']):
            self.vectorizers['title'] = loader.load_vectorizer('MiniLM', model_type='sentence_transformer')

        # Load About Vectorizer
        if not ('about' in self.vectorizers and self.vectorizers['about']):
            self.vectorizers['about'] = loader.load_vectorizer('jobs_tfidf.pkl', model_type='sklearn')

        # Load Preferred Work Types Encoder
        if not ('preferred_work_types' in self.encoders and self.encoders['preferred_work_types']):
            self.encoders['preferred_work_types'] = loader.load_encoder('work_type_onehot_enc.pkl', model_type='sklearn')

    def preprocess(self, obj: user):
        # Clean the title and about
        obj.title = self.preprocessor.clean(obj.title)
        obj.about = self.preprocessor.clean(obj.about)
        
        # Refactor the work types as a list of lists
        obj.preferred_work_types = [[work_type] for work_type in obj.preferred_work_types]
        
        
        return obj
