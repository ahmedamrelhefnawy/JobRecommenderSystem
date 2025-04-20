import os

workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'

URL_PATTERN = r'(?:http[s]?:\/\/.)?(?:www\.)?[-a-zA-Z0-9@%._\+~#=]{2,256}\.[a-z]{2,6}\b(?:[-a-zA-Z0-9@:%_\+.~#?&\/\/=]*)'

job_recommendation_features = ['title', 'content', 'work_type']