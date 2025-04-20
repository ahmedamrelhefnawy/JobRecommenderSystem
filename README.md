# Job Recommender System

This project is a Job Recommender System that matches user profiles with job descriptions using advanced natural language processing (NLP) techniques.

## Features

- Content-based job recommendation
- Uses lightweight models for text representation and similarity calculation
- Utilizes Sentence Transformers for title vectorization
- Uses TF-IDF for content vectorization
- Supports multiple features including title, content, work type, and skills
- Parallel processing for efficient candidate ranking
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

This project is licensed under the [GNU GPLv3](LICENSE).