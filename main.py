from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

print("Loading the recommender...")
from Modules.recommender import job_recommender
cb_recommender = job_recommender()
print("Recommender loaded successfully")

class user(BaseModel    ):
    title: str
    about: str
    preferred_work_types: list[str] | None
    experience_level: str | None
    expected_salary: int | None
    skills: list[str] | None

class job(BaseModel):
    title: str
    content: str
    work_type: str | None

class weights(BaseModel):
    title: float = 0.2
    content: float = 0.5
    work_type: float = 0.1
    skills: float = 0.2

@app.post("/recommend/user", description="Recommend jobs based on user information")
async def user_based_recommend(user: user = Body(..., title="User data", description="User data in JSON format"),
                                jobs: list[job] = Body(..., title="Jobs data", description="Jobs data in JSON format"),
                                weights: weights = Body(weights(), title="Score Evaluation Weights")) -> JSONResponse:
    
    user = user.model_dump()
    jobs = [job.model_dump() for job in jobs]
    weights = weights.model_dump()
    
    recommendations = cb_recommender.user_job_recommend(user, jobs, weights)
    
    return JSONResponse(content= {'recommendations': recommendations})

@app.post("/recommend/job", description="Recommend jobs based on a job information")
async def job_based_recommend(base_job: job = Body(..., title="Base Job data", description="Base Job data in JSON format"),
                            jobs: list[job] = Body(..., title="Jobs data", description="Jobs data in JSON format"),
                            weights: weights = Body(weights(), title="Score Evaluation Weights")) -> JSONResponse:
    
    base_job = base_job.model_dump()
    jobs = [job.model_dump() for job in jobs]
    weights = weights.model_dump()
    
    recommendations = cb_recommender.job_recommend(base_job, jobs, weights)

    return JSONResponse(content= {'recommendations': recommendations})
