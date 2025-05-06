from pydantic import BaseModel, Field


class user(BaseModel):
    """
    User model representing a job seeker's profile.

    Attributes:
        user_id (int): Unique identifier for the user
        title (str): User's professional title or desired job title
        about (str): User's professional summary or description
        preferred_work_types (list[str] | None): List of preferred work arrangements
            Valid values: ["FULL_TIME", "PART_TIME", "CONTRACT", "INTERNSHIP", "REMOTE"]
        experience_level (str | None): User's experience level
            Valid values: ["Entry level", "Mid level", "Senior", "Executive"]
        expected_salary (int | None): Expected annual salary in the local currency
        skills (list[str] | None): List of user's technical and professional skills
    """
    user_id: int = Field(..., gt=0)
    title: str = Field(..., min_length=1)
    about: str = Field(..., min_length=1)
    preferred_work_types: list[str] | None
    experience_level: str | None
    expected_salary: int | None
    skills: list[str] | None


class job(BaseModel):
    """
    Job model representing a job posting.

    Attributes:
        job_id (int): Unique identifier for the job posting
        title (str): Job position title
        content (str): Full job description including requirements and responsibilities
        work_type (str | None): Type of work arrangement
            Valid values: ["FULL_TIME", "PART_TIME", "CONTRACT", "INTERNSHIP", "REMOTE"]
    """
    job_id: int = Field(..., gt=0)
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    work_type: str | None = Field(None, pattern="^(FULL_TIME|PART_TIME|CONTRACT|INTERNSHIP|REMOTE)$")


class weights(BaseModel):
    """
    Weights model for configuring the importance of different factors in job recommendations.

    Attributes:
        title (float): Weight for job title similarity
        content (float): Weight for job description/content similarity
        work_type (float): Weight for work type matching
        skills (float): Weight for skills matching
        
    Note: The sum of all weights should equal 1.0
    """
    title: float = 0.4
    content: float = 0.5
    work_type: float = 0.1