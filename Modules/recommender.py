import numpy as np
from abc import ABC

from Modules.model_handlers import model_loader, embed
from Modules import utils
from Models.models import weights
from Modules.database import EmbeddingDB
from Modules import consts
loader = model_loader()
class content_based_recommender(ABC):
    def __init__(self, db: EmbeddingDB):
        self.db = db

    
    def _recommend(self, base: dict, options: dict, recommender_weights: weights):    
        similarity = dict()

        # Calculate options similarity for each feature
        for feature in consts.job_recommendation_features:
            # Add feature similarity to similarity dictionary
            similarity[feature] = np.dot(base[feature], options[feature].T).squeeze()
        
        # Get the number of options
        temp_feature = list(similarity.keys())[0]
        n_options = len(similarity[temp_feature])
        
        # Initialize option scores array of zeros
        option_scores = np.zeros((n_options, ), dtype= np.float64)
        
        # Accumulate scores
        for feature, scores in similarity.items():
            feature_weight = getattr(recommender_weights, feature)
            option_scores += feature_weight * utils.normalize(scores)

        # Sort options by score
        sorted_options = sorted(enumerate(option_scores), key= lambda x: x[1], reverse= True)
        return sorted_options
    
class job_recommender(content_based_recommender):
    
    def job_recommend(self, base_job_id: int, jobs_ids: list[int], recommender_weights: weights):
        
        # Get embeddings from database
        base_job = self.db.get_job_embeddings(base_job_id)
        jobs = self._fetch_jobs(jobs_ids)

        # Get recommendations
        recommendations = self._recommend(base_job, jobs, recommender_weights)
        
        # Get recommendations ids
        recommendations = self._get_recommendations_ids(jobs_ids, recommendations)
        return recommendations
    
    
    def user_job_recommend(self, user_id: int, jobs_ids: list[int], recommender_weights: weights):
        # Get user embeddings from database
        user = self.db.get_user_embeddings(user_id)
        
        # Map user keys to job keys 
        user = self._user_job_map(user)
        
        
        # Get jobs embeddings from database
        jobs = self._fetch_jobs(jobs_ids)
        
        # Get recommendations
        recommendations = self._recommend(user, jobs, recommender_weights)
        
        # Get recommendations ids
        recommendations = self._get_recommendations_ids(jobs_ids, recommendations)
        
        return recommendations

    def _get_recommendations_ids(self, jobs_ids: list[int], recommendations: list[tuple[int, float]]):
        return [(jobs_ids[i], score) for i, score in recommendations]
    
    def _fetch_jobs(self, jobs_ids: list[int]):
        # Define jobs dictionary to store each job's feature embeddings
        jobs = dict()
        
        for feature in consts.job_recommendation_features:
            # Get jobs feature embeddings column from database
            embeddings = self.db.get_jobs_column_embeddings(jobs_ids, feature)
            jobs[feature] = embeddings
        
        return jobs
    
    def _user_job_map(self, user_data: dict):
        # Rearrange the work types as a list of lists
        user_data['preferred_work_types'] = [[work_type] for work_type in user_data['preferred_work_types']]
        
        # Map user keys to recommender keys
        user_map = {
                    'about': 'content',
                    'preferred_work_types':'work_type',
                    'Expected Salary': 'normalized_salary'
                    }
        
        # Map user keys to job keys
        for key, value in user_map.items():
            utils.rename_key(user_data, key, value)
        
        return user_data