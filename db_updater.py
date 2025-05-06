from Models.models import user, job
from dotenv import load_dotenv
import time
import gc

load_dotenv()

def process_unflagged_jobs(job_embed, db, msdb):
    # Fetch unflagged jobs
    unflagged_jobs = msdb.get_unflagged_jobs()
    
    if unflagged_jobs:
        # Extract job IDs and create job objects
        job_ids = []
        jobs_data = []
        for jb in unflagged_jobs:
            job_ids .append(jb['JobId'])

            # Create job object
            try:
                job_object = job(
                    job_id=jb['JobId'],
                    title=jb['JobTitle'],
                    content=jb['Description'],
                    work_type=jb['JobType'],
                    )
                jobs_data.append(job_object)
            except Exception as e:
                print(f"Error creating job object for ID {jb['JobId']}")
        
        # Generate embeddings for all jobs
        if jobs_data:
            job_embeddings = job_embed.embed_batch(jobs_data)

            # Store embeddings in database for each job
            for job_id, job_embeddings in zip(job_ids, job_embeddings):
                db.store_job_embeddings(job_id, job_embeddings)
                del job_embeddings  # Delete job embeddings after storing

            msdb.flag_jobs(job_ids)  # Mark job as processed
            del job_embeddings  # Free memory
    
    del unflagged_jobs


def process_unflagged_users(user_embed, db, msdb):
    # F etch unflagged users
    unflagged_users = msdb.get_unflagged_users()
    
    if unflagged_users:
        # Start Embedding Process
        # Extract user IDs and create user objects
        user_ids = []
        users_data = []
        for usr in unflagged_users:
            user_ids.append(usr['Id'])

            # Create user object
            try:
                usr_object = user(
                    user_id=usr['Id'],
                    title=usr['Title'],
                    about=usr['About'],
                    preferred_work_types=[usr['JobTypePreference']],
                    )
                users_data.append(usr_object)
            except Exception as e:
                print(f"Error creating user object for ID {usr['Id']}")
        
        # Generate embeddings for all users
        if users_data:
            
            users_batch_embeddings = user_embed.embed_batch(users_data)

            # Store embeddings in database for each user
            for user_id, user_embeddings in zip(user_ids, users_batch_embeddings):
                db.store_user_embeddings(user_id, user_embeddings)
                del user_embeddings  # Delete individual embeddings after storing

            msdb.flag_users(user_ids)  # Mark user as processed
            del users_batch_embeddings  # Free memory
    
    del unflagged_users

def process_unflagged_entities(user_embed, job_embed, db, msdb, period=60):
    while True:
        print("Processing unflagged entities...")
        try:
            # Process unflagged users
            process_unflagged_users(user_embed, db, msdb)

            # Process unflagged jobs
            process_unflagged_jobs(job_embed, db, msdb)

            # Free memory
            gc.collect()
            
        except Exception as e:
            print(f"Error processing unflagged entities: {e}")

        # Wait for 1 minute before the next check
        print(f"Sleeping for {period} seconds...")
        time.sleep(period)

if __name__ == "__main__":
    # Example usage
    # Initialize database
    print("Loading the database...")
    from Modules.database import EmbeddingDB, ServerDB
    db = EmbeddingDB()
    msdb = ServerDB()
    print("Database loaded successfully")

    # Initialize embedders
    print("Loading the embedders...")
    from Modules.preprocessor import job_embedder, user_embedder
    user_embed = user_embedder()
    job_embed = job_embedder()
    print("Embedders loaded successfully")


    # Start the process in a separate thread or process
    process_unflagged_entities(user_embed, job_embed, db, msdb, period=60)