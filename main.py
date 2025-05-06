from Modules.utils import filter_recommendations
from Models.models import user, job, weights

from fastapi import FastAPI, Body, HTTPException, Request
from fastapi.responses import JSONResponse

from typing import List, Optional
import gc


# Initialize database
print("Loading the database...")
from Modules.database import EmbeddingDB, ServerDB
db = EmbeddingDB()
msdb = ServerDB()
print("Database loaded successfully")

# Initialize recommender
print("Loading the recommender...")
from Modules.recommender import job_recommender
recommender = job_recommender(db)
print("Recommender loaded successfully")

app = FastAPI()

@app.middleware("http")
async def cleanup_after_request(request: Request, call_next):
    response = await call_next(request)
    gc.collect()
    return response


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
        
        return JSONResponse(content={'user_id': user_id, 'recommendations': filtered_recommendations})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/recommend/job", description="Recommend jobs based on a job information")
async def job_based_recommend(
    base_job_id: int = Body(..., title="Base Job ID", description="ID of the base job to get recommendations for", gt=0),
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

        return JSONResponse(content={'base_job_id': base_job_id, 'recommendations': filtered_recommendations})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e