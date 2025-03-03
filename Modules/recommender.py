import numpy as np
from abc import ABC, abstractmethod
from .model_handlers import model_loader, embed
from . import utils


loader = model_loader()

class content_based_recommender(ABC):
    def __init__(self,
                vectorizers: dict= None,
                encoders: dict   = None
                ):
        self.vectorizers = vectorizers
        self.encoders = encoders
        
        # Handle missing vectorizers
        if not self.vectorizers:
            self.vectorizers = dict()
            
        # Handle missing encoders
        if not self.encoders:
            self.encoders = dict()
    
    def _recommend(self, base: dict, options: list[dict], weights: dict):
        similarity = dict()

        for feature in base.keys():
            
            if feature in self.vectorizers: # Feature is vectorizable text
                base_embd = embed(self.vectorizers[feature], [base[feature]])
                options_embd = embed(self.vectorizers[feature], [option[feature] for option in options])
                
            elif feature in self.encoders: # Feature is categorical
                base_embd = embed(self.encoders[feature], base[feature])
                base_embd = np.sum(base_embd, axis= 0)
                
                options_embd = embed(self.encoders[feature], [[option[feature]] for option in options])
                
            else: # Feature not supported
                continue
            
            # Add feature Similarity
            similarity[feature] = np.dot(base_embd, options_embd.T).squeeze()
        
        # Accumulate Scores
        n_options = len(options)
        option_scores = np.zeros((n_options, ), dtype= np.float64)
        
        for feature, scores in similarity.items():
            option_scores += weights[feature] * utils.normalize(scores)

        sorted_options = sorted(enumerate(option_scores), key= lambda x: x[1], reverse= True)
        return sorted_options
    
class job_recommender(content_based_recommender):
    def __init__(self,
                vectorizers: dict = None,
                encoders: dict = None
                ):
        super().__init__(vectorizers, encoders)
        
        # Load Title Vectorizer
        if not ('title' in self.vectorizers and self.vectorizers['title']):
            self.vectorizers['title'] = loader.load_vectorizer('MiniLM', type= 'sentence_transformer')
        
        # Load Content Vectorizer
        if not ('content' in self.vectorizers and self.vectorizers['content']):
            self.vectorizers['content'] = loader.load_vectorizer('jobs_tfidf.pkl', type='sklearn')
        
        # Load Encoders
        if not ('work_type' in self.encoders and self.encoders['work_type']):
            self.encoders['work_type'] = loader.load_encoder('work_type_onehot_enc.pkl', type= 'sklearn')
        
        
        # Check for intersection between vectorizers and encoders
        intersect_keys = set(self.vectorizers.keys()).intersection(self.encoders.keys())
        if intersect_keys:
            raise ValueError(f"Can't Assign two models to the same feature\nConflict with features: {intersect_keys}")
    
    
    def job_recommend(self, base_job: dict, jobs: list[dict], weights: dict):
        base_job['work_type'] = [[base_job['work_type']]]
        
        return self._recommend(base_job, jobs, weights)
    
    
    def user_job_recommend(self, user: dict, jobs: list[dict], weights: dict):
        # Rearrange the work types as a list of lists
        user['preferred_work_types'] = [[work_type] for work_type in user['preferred_work_types']]
        
        # Map user keys to recommender keys
        user_map = {
                    'about': 'content',
                    'preferred_work_types':'work_type',
                    'Expected Salary': 'normalized_salary'
                    }
        
        # Map user keys to job keys
        for key, value in user_map.items():
            utils.rename_key(user, key, value)

        return self._recommend(user, jobs, weights)