from Modules.utils import filter_recommendations
from fastapi import FastAPI, Body, HTTPException, Request
from fastapi.responses import JSONResponse
from Models.models import user, job, weights
from typing import List, Optional
import gc

app = FastAPI()

# Initialize database
print("Loading the database...")
from Modules.database import EmbeddingDB
db = EmbeddingDB()
print("Database loaded successfully")

# Initialize embedders
print("Loading the embedders...")
from Modules.recommender import job_recommender
from Modules.preprocessor import job_embedder, user_embedder
job_embed = job_embedder()
user_embed = user_embedder()
print("Embedders loaded successfully")

# Initialize recommender
print("Loading the recommender...")
recommender = job_recommender(db)
print("Recommender loaded successfully")

@app.middleware("http")
async def cleanup_after_request(request: Request, call_next):
    response = await call_next(request)
    gc.collect()
    return response

@app.post("/add_user/", description="Add a new user to the database")
async def add_user(user_data: user = Body(..., title="User data", description="User data in JSON format")) -> JSONResponse:
    try:
        user_id = user_data.user_id

        # Generate embeddings for user data
        user_embeddings = user_embed.embed(user_data)

        # Store embeddings in database
        db.store_user_embeddings(user_id, user_embeddings)

        # Explicitly delete large objects
        del user_embeddings

        return JSONResponse(content={"message": "User created successfully", "user_id": user_id})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/add_job/", description="Add a new job to the database")
async def add_job(job_data: job = Body(..., title="Job data", description="Job data in JSON format")) -> JSONResponse:
    try:
        job_id = job_data.job_id

        # Generate embeddings for job data
        job_embeddings = job_embed.embed(job_data)

        # Store embeddings in database
        db.store_job_embeddings(job_id, job_embeddings)

        # Explicitly delete large objects
        del job_embeddings

        return JSONResponse(content={"message": "Job created successfully", "job_id": job_id})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/add_users_batch/", description="Add multiple users to the database")
async def add_users_batch(users_data: List[user] = Body(..., title="Users data", description="List of users data in JSON format")) -> JSONResponse:
    try:
        user_ids = [user_data.user_id for user_data in users_data]

        # Generate embeddings for all users
        users_embeddings = user_embed.embed_batch(users_data)

        # Store embeddings in database for each user
        for user_id, user_embeddings in zip(user_ids, users_embeddings):
            db.store_user_embeddings(user_id, user_embeddings)
            del user_embeddings  # Delete individual embeddings after storing

        # Explicitly delete large objects
        del users_embeddings

        return JSONResponse(content={
            "message": "Users batch created successfully",
            "user_ids": user_ids
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/add_jobs_batch/", description="Add multiple jobs to the database")
async def add_jobs_batch(jobs_data: List[job] = Body(..., title="Jobs data", description="List of jobs data in JSON format")) -> JSONResponse:
    try:
        job_ids = [job_data.job_id for job_data in jobs_data]

        # Generate embeddings for all jobs
        jobs_embeddings = job_embed.embed_batch(jobs_data)

        # Store embeddings in database for each job
        for job_id, job_embeddings in zip(job_ids, jobs_embeddings):
            db.store_job_embeddings(job_id, job_embeddings)
            del job_embeddings  # Delete individual embeddings after storing

        # Explicitly delete large objects
        del jobs_embeddings

        return JSONResponse(content={
            "message": "Jobs batch created successfully",
            "job_ids": job_ids
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/recommend/user", description="Recommend jobs based on user information")
async def user_based_recommend(
    user_id: int = Body(..., title="User ID", description="ID of the user to get recommendations for"),
    jobs_ids: list[int] = Body(..., title="Jobs IDs", description="IDs of the jobs to get recommendations for"),
    recommender_weights: weights = Body(weights(), title="Score Evaluation Weights"),
    max_recommendations: int = Body(10, title="Maximum number of recommendations", description="Maximum number of recommendations to return"),
    threshold: Optional[float] = Body(None, title="Threshold for recommendations", description="Minimum score for recommendations\nNote: if provided, max_recommendations will be ignored")
) -> JSONResponse:
    try:
        # Check if user exists
        missing_user_id = db.get_missing_user_ids([user_id])[0]
        if missing_user_id:
            return JSONResponse(
                status_code=404,
                content={"error": "User not found",
                         "missing_user_id": missing_user_id}
            )
        
        # Get missing job ids
        missing_job_ids = db.get_missing_job_ids(jobs_ids)
        if missing_job_ids:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "Jobs not found",
                    "missing_job_ids": missing_job_ids
                }
            )
        
        # Get recommendations
        recommendations = recommender.user_job_recommend(user_id, jobs_ids, recommender_weights)

        # Filter recommendations
        filtered_recommendations = filter_recommendations(recommendations, max_recommendations, threshold)
        
        return JSONResponse(content={'recommendations': filtered_recommendations})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/recommend/job", description="Recommend jobs based on a job information")
async def job_based_recommend(
    base_job_id: int = Body(..., title="Base Job ID", description="ID of the base job to get recommendations for"),
    jobs_ids: list[int] = Body(..., title="Jobs IDs", description="IDs of the jobs to get recommendations for"),
    recommender_weights: weights = Body(weights(), title="Score Evaluation Weights"),
    max_recommendations: int = Body(10, title="Maximum number of recommendations", description="Maximum number of recommendations to return"),
    threshold: Optional[float] = Body(None, title="Threshold for recommendations", description="Minimum score for recommendations\nNote: if provided, max_recommendations will be ignored")
) -> JSONResponse:
    try:
        # Check if base job exists
        missing_base_job_id = db.get_missing_job_ids([base_job_id])[0]
        if missing_base_job_id:
            return JSONResponse(
                status_code=404,
                content={"error": "Base job not found",
                         "missing_base_job_id": missing_base_job_id}
            )

        # Get missing job ids
        missing_job_ids = db.get_missing_job_ids(jobs_ids)
        if missing_job_ids:
            return JSONResponse(
                status_code=404,
                content={
                    "error": "Jobs not found",
                    "missing_job_ids": missing_job_ids
                }
            )

        # Get recommendations
        recommendations = recommender.job_recommend(base_job_id, jobs_ids, recommender_weights)
        
        # Filter recommendations
        filtered_recommendations = filter_recommendations(recommendations, max_recommendations, threshold)

        return JSONResponse(content={'recommendations': filtered_recommendations})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e