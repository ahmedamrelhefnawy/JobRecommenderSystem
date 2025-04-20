import os
import pickle
from .consts import workspace_dir
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

models_path = workspace_dir + 'AI_Models/'
vectorizers_path = models_path + 'Vectorizers/'
encoders_path = models_path + 'Encoders/'


def embed(model, inpt):
    if isinstance(model, SentenceTransformer):
        return model.encode(inpt)

    elif isinstance(model, TfidfVectorizer):
        return model.transform(inpt).toarray()

    elif isinstance(model, OneHotEncoder):
        return model.transform(inpt).toarray()
    else:
        raise ValueError('Invalid model type:', type(model))


class model_loader:

    def __load_model(self, path: str):

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, 'rb') as f:
            model = pickle.load(f)

        return model

    def load_sentence_transformer(self, path: str):
        return SentenceTransformer(path)

    def load_sklearn_model(self, path: str):
        return self.__load_model(path)

    def load_vectorizer(self, name: str, model_type: str):
        if model_type == 'sentence_transformer':
            return self.load_sentence_transformer(vectorizers_path + name)
        elif model_type == 'sklearn':
            return self.load_sklearn_model(vectorizers_path + name)
        else:
            raise ValueError('Invalid vectorizer type:', model_type)

    def load_encoder(self, name: str, model_type: str):
        if model_type == 'sklearn':
            return self.load_sklearn_model(encoders_path + name)
        else:
            raise ValueError('Invalid encoder type:', model_type)
