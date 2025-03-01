# Job Recommender System

This project is a Job Recommender System that matches user profiles with job descriptions using advanced natural language processing (NLP) techniques.

## Features

- Content-based job recommendation
- Uses Light weight models for text representation and similarity calculation
- Utilizes Sentence Transformers for title vectorization
- Uses TF-IDF for content vectorization
- Supports multiple features including title, content, work type, and skills
- Parallel processing for efficient candidate ranking

## Installation

1. Clone the repository:
    ```sh
    git clone <repository-url>
    ```

2. Navigate to the project directory:
    ```sh
    cd "Job Recommender System"
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Input Format

#### User Profile

The user profile should be a dictionary with the following keys:

- `title`: Job title (string)
- `about`: Description about the user (string)
- `preferred_work_types`: List of preferred work types (list of strings)
- `experience_level`: Experience level (string)
- `expected_salary`: Expected salary (optional, can be `None`)
- `skills`: List of skills (list of strings)

Example:
```python
user = {
    'title': 'Machine Learning Engineer',
    'about': 'Machine Learning Engineer with a strong focus on building intelligent systems...',
    'preferred_work_types': ['FULL_TIME', 'INTERNSHIP'],
    'experience_level': 'Entry level',
    'expected_salary': None,
    'skills': ['python', 'scikitlearn', 'tensorflow' 'pandas']
}
```

#### Jobs

The job descriptions should be a list of dictionaries, each containing the following keys:

- `title`: Job title (string)
- `description`: Job description (string)
- `work_type`: Work type (string)
- `skills`: List of required skills (list of strings)

Example:
```python
jobs = [
    {
        'title': 'Data Scientist',
        'description': 'We are looking for a Data Scientist to analyze large amounts of raw information...',
        'work_type': 'FULL_TIME',
        'skills': ['python', 'machine learning', 'data analysis']
    },
    ...
]
```
---
### Default Used NLP Models
#### Title vectorization:
Library: **Sentence Transformers**
Model: **sentence-transformers/all-MiniLM-L6-v2**
```python
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
```

#### Content (Description) vectorization:
Library: **Scikit-Learn**
Method: **TF-IDF**
```python
    TfidfVectorizer(max_df= 0.95, min_df= 0.0001, stop_words='english')
```

---
### Running the Recommender

1. Load the recommender:
    ```python
    from Modules.recommender import content_based_job_recommender as cbjr
    recommender = cbjr()
    ```

2. Define the weights for different features:
    ```python
    weights = {
        'title': 0.2,
        'content': 0.5,
        'work_type': 0.1,
        'skills': 0.2
    }
    ```

3. Get job recommendations for the user:
    ```python
    recommendations = recommender.user_job_recommend(user, jobs, weights)
    ```

4. Print the top 5 recommendations:
    ```python
    for idx, score in recommendations[:5]:
        job = jobs[idx]
        print(f"Job Title: {job['title']}")
        print(f"Description: {job['description'][:50]}...")
        print(f"Score: {score}\n")
    ```
---
This project is licensed under the [GNU GPLv3](LICENSE).