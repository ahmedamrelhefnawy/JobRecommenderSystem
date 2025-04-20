# Job Recommender System

This project is a Job Recommender System that matches user profiles with job descriptions using advanced natural language processing (NLP) techniques.

## Features

- Content-based job recommendation
- Uses lightweight models for text representation and similarity calculation
- Uses lightweight models for text representation and similarity calculation
- Utilizes Sentence Transformers for title vectorization
- Uses TF-IDF for content vectorization
- Supports multiple features including title, content, work type, and skills
- Parallel processing for efficient candidate ranking
- RESTful API for easy integration
- RESTful API for easy integration

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

## API Documentation

The Job Recommender System provides a RESTful API with the following endpoints:

### Data Models

#### User Model
- `user_id` (integer): Unique identifier for the user
- `title` (string): User's professional title or desired job title
- `about` (string): User's professional summary or description
- `preferred_work_types` (array[string] | null): List of preferred work arrangements
  - Valid values: ["FULL_TIME", "PART_TIME", "CONTRACT", "INTERNSHIP", "REMOTE"]
- `experience_level` (string | null): User's experience level
  - Valid values: ["Entry level", "Mid level", "Senior", "Executive"]
- `expected_salary` (integer | null): Expected annual salary in the local currency
- `skills` (array[string] | null): List of user's technical and professional skills

#### Job Model
- `job_id` (integer): Unique identifier for the job posting
- `title` (string): Job position title
- `content` (string): Full job description including requirements and responsibilities
- `work_type` (string | null): Type of work arrangement
  - Valid values: ["FULL_TIME", "PART_TIME", "CONTRACT", "INTERNSHIP", "REMOTE"]

#### Weights Model
- `title` (float): Weight for job title similarity
- `content` (float): Weight for job description/content similarity
- `work_type` (float): Weight for work type matching
- `skills` (float): Weight for skills matching

Note: The sum of all weights should equal 1.0

### Endpoints

#### Add User
```http
POST /add_user/
```
Adds a new user to the database.

**Request Body:**
```json
{
    "user_id": integer,
    "title": string,
    "about": string,
    "preferred_work_types": array[string],
    "experience_level": string,
    "expected_salary": integer | null,
    "skills": array[string]
}
```

**Response:**
```json
{
    "message": "User created successfully",
    "user_id": integer
}
```

#### Add Batch of Users at once
```http
POST /add_users_batch/
```
Adds multiple users to the database in a single request.

**Request Body:**
```json
[
    {
        "user_id": integer,
        "title": string,
        "about": string,
        "preferred_work_types": array[string],
        "experience_level": string,
        "expected_salary": integer | null,
        "skills": array[string]
    }
]
```

**Response:**
```json
{
    "message": "Users batch created successfully",
    "user_ids": array[integer]
}
```

#### Add Job
```http
POST /add_job/
```
Adds a new job to the database.

**Request Body:**
```json
{
    "job_id": integer,
    "title": string,
    "content": string,
    "work_type": string | null
}
```

**Response:**
```json
{
    "message": "Job created successfully",
    "job_id": integer
}
```

#### Add Batch of Jobs at once
```http
POST /add_jobs_batch/
```
Adds multiple jobs to the database in a single request.

**Request Body:**
```json
[
    {
        "job_id": integer,
        "title": string,
        "content": string,
        "work_type": string | null
    }
]
```

**Response:**
```json
{
    "message": "Jobs batch created successfully",
    "job_ids": array[integer]
}
```

#### User-Based Job Recommendations
```http
POST /recommend/user
```
Get job recommendations based on a user's profile.

**Request Body:**
```json
{
    "user_id": integer,
    "jobs_ids": array[integer],
    "recommender_weights": {
        "title": float,
        "content": float,
        "work_type": float,
        "skills": float
    },
    "max_recommendations": integer (default: 10),
    "threshold": float | null
}
```

**Note:** If `threshold` is provided, `max_recommendations` will be ignored and all recommendations with scores above the threshold will be returned.

**Response:**
```json
{
    "recommendations": [
        {
            "job_id": integer
        }
    ]
}
```

#### Job-Based Recommendations
```http
POST /recommend/job
```
Get similar job recommendations based on a reference job.

**Request Body:**
```json
{
    "base_job_id": integer,
    "jobs_ids": array[integer],
    "recommender_weights": {
        "title": float,
        "content": float,
        "work_type": float,
        "skills": float
    },
    "max_recommendations": integer (default: 10),
    "threshold": float | null
}
```

**Note:** If `threshold` is provided, `max_recommendations` will be ignored and all recommendations with scores above the threshold will be returned.

**Response:**
```json
{
    "recommendations": [
        {
            "job_id": integer
        }
    ]
}
```

### Error Responses
All endpoints may return the following error responses:

- `404 Not Found`: When a requested resource (user or job) is not found
- `500 Internal Server Error`: When an unexpected error occurs during processing

## Technical Details

### NLP Models Used

#### Title Vectorization
- Library: **Sentence Transformers**
- Model: **sentence-transformers/all-MiniLM-L6-v2**
```python
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
```

#### Content Vectorization
- Library: **Scikit-Learn**
- Method: **TF-IDF**

```python
TfidfVectorizer(max_df=0.95, min_df=0.0001, stop_words='english')
```


## License

## License

This project is licensed under the [GNU GPLv3](LICENSE).