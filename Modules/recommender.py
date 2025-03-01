import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

from .consts import workspace_dir
from . import utils

models_dir = workspace_dir + '/Models'

def load_encoder(path):
    with open(path, 'rb') as f:
        encoder = pickle.load(f)
    return encoder

def load_vectorizer(path):
    with open(path, 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer

class content_based_job_recommender:
    def __init__(self,
                title_vectorizer = None,
                content_vectorizer = None,
                encoders= {
                    'work_type': load_encoder(models_dir + '/Encoders/work_type_onehot_enc.pkl')
                    }
                ):
        
        if not title_vectorizer:
            title_vectorizer = SentenceTransformer(models_dir + '/Vectorizers/MiniLM/')
        self.title_vectorizer = title_vectorizer
        
        if not content_vectorizer:
            content_vectorizer = load_vectorizer(models_dir + '/Vectorizers/tfidf.pkl')
        self.content_vectorizer = content_vectorizer
        
        self.encoders = encoders
        
        self.user_job_map = {
                                'about': 'description',
                                'preferred_work_types':'work_type',
                                'experience_level': 'formatted_experience_level',
                                'Expected Salary': 'normalized_salary'
                            }
    
    def get_matches_score(self, ind1, ind1_param, ind2, ind2_param):
        encoder = self.encoders[ind1_param]

        n_categs = len(encoder.categories_)

        ind1_enc = np.zeros(shape= (1, n_categs), dtype= np.int32)
        for option in ind1[ind1_param]:
            ind1_enc = ind1_enc + encoder.transform([[option]]).toarray()

        ind2_enc = encoder.transform([[ind[ind2_param]] for ind in ind2]).toarray()

        # Calculate similarity
        scores = np.dot(ind1_enc, ind2_enc.T)[0]
        return scores

    def job_recommend(self, base_job: dict, jobs: list[dict], weights: dict):
        title_vectorizer = self.title_vectorizer
        content_vectorizer = self.content_vectorizer

        similarity = dict()
        if weights['title']:
            # Get Title Embeddings
            base_job_title = title_vectorizer.encode([base_job['title']])
            jobs_title  = title_vectorizer.encode([job['title'] for job in jobs])

            # Calculate Title Similarity
            similarity['title'] = np.dot(base_job_title, jobs_title.T)[0]

        if weights['content']:
            # Get Description (Content) Embeddings
            base_job_content_embd = content_vectorizer.transform([base_job['description']]).toarray()
            jobs_content_embd = content_vectorizer.transform([job['description'] for job in jobs]).toarray()

            # Calculate Content Embeddings Similarity
            similarity['content'] = np.dot(base_job_content_embd, jobs_content_embd.T)[0]

        # Calculate Work Type Similarity
        if 'work_type' in base_job.keys() and weights['work_type']:
            similarity['work_type'] = self.get_matches_score(base_job, 'work_type', jobs, 'work_type')
        else:
            similarity['work_type'] = 0

        # Get Skill Match
        if 'skills' in jobs[0] and weights['skills']:
            base_job_skill_embd = content_vectorizer.transform(base_job['skills']).toarray()
            jobs_skill_embd = content_vectorizer.transform([job['skills'] for job in jobs]).toarray()

            temp_similarity = np.dot(base_job_skill_embd, jobs_skill_embd.T)
            similarity['skills'] = np.average(temp_similarity)
        
        # Accumulate Scores
        n_jobs = len(jobs)
        job_scores = np.zeros((n_jobs, ), dtype= np.float64)
        
        for feature, scores in similarity.items():
            job_scores += weights[feature] * utils.normalize(scores)

        job_scores = sorted(enumerate(job_scores), key= lambda x: x[1], reverse= True)
        return job_scores
    
    def user_job_recommend(self, user: dict, jobs: list[dict], weights: dict):
        # Map user keys to job keys
        for key, value in self.user_job_map.items():
            utils.rename_key(user, key, value)

        return self.job_recommend(user, jobs, weights)
