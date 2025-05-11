
from Modules.recommender import job_recommender
from Modules.preprocessor import user_embedder, job_embedder
from Modules.utils import filter_recommendations
from Models.models import user, job, weights
from Modules.database import EmbeddingDB
import pandas as pd
import pickle
import torch
from tqdm import tqdm
import streamlit as st

torch.classes.__path__ = []


if 'jobs' not in st.session_state:
    st.session_state.jobs = None
if 'users' not in st.session_state:
    st.session_state.users = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'weights' not in st.session_state:
    st.session_state.weights = weights()

if 'user_embed' not in st.session_state:
    print("Loading User Embedder ...")
    st.session_state.user_embed = user_embedder()
    print("Loaded User Embedder Successfully")

if 'job_embed' not in st.session_state:
    print("Loading Job Embedder ...")
    st.session_state.job_embed = job_embedder()
    print("Loaded Job Embedder Successfully")

if 'db' not in st.session_state:
    print("Loading Database ...")
    st.session_state.db = EmbeddingDB()
    print("Loaded Database Successfully")

if 'recommender' not in st.session_state:
    print("Loading Recommender ...")
    st.session_state.recommender = job_recommender(st.session_state.db)
    print("Loaded Recommender Successfully")

user_embed = st.session_state.user_embed
job_embed = st.session_state.job_embed
db = st.session_state.db
recommender = st.session_state.recommender

if st.session_state.jobs is None:
    with st.spinner("Loading jobs..."):
        print("Loading jobs ...")
        jobs: pd.DataFrame = pickle.load(open('/media/ahmedelhefnawy/NewVolume/DEPI/Mega Project/Job Recommender System/test_data.pkl', 'rb'))
        print("Jobs Loaded successfully")

        ids = [i for i in range(len(jobs))]

        job_objs = [None] * len(ids)
        for i, _id in enumerate(ids):
            jb = jobs[_id]
            job_objs[i] = job(
                job_id=i,
                title=jb['title'],
                content=jb['description'],
                work_type=jb['work_type']
            )


        missing_ids = db.get_missing_job_ids(ids)
        if missing_ids:
            for _id in tqdm(missing_ids):
                jb = job_objs[_id]
                embeddings = job_embed.embed(jb.model_copy())
                db.store_job_embeddings(jb.job_id, embeddings)
        
        st.session_state.jobs = job_objs
        print("Jobs stored successfully")

if st.session_state.users is None:
    with st.sidebar and st.spinner("Loading users..."):
        user1 = {'title': 'Machine Learning Engineer',
                    'about': '''Machine Learning Engineer with a strong focus on building intelligent systems that bridge the gap between user needs and actionable insights. Currently, I am working on developing a sophisticated recommender system designed to match user profiles with job descriptions, leveraging advanced natural language processing (NLP) and collaborative filtering techniques. My role involves end-to-end ownership of the project, from data preprocessing and feature engineering to model training, evaluation, and deployment.

        With a solid foundation in Computer Science, I bring 4 years of experience in designing and implementing machine learning solutions that drive real-world impact. My expertise includes working with large-scale datasets, optimizing recommendation algorithms, and deploying models into production environments using cloud platforms like AWS and GCP. I am proficient in Python, TensorFlow, PyTorch, and Scikit-learn, and have hands-on experience with tools like Spark and Docker for scalable data processing and deployment.

        In my current project, I am focused on improving the accuracy and personalization of job recommendations by incorporating user behavior data, contextual information, and advanced NLP techniques to better understand both user profiles and job descriptions. This involves experimenting with deep learning architectures, such as BERT-based models, and continuously iterating to enhance system performance. For example, I recently implemented a hybrid recommendation approach that combines content-based filtering with matrix factorization, resulting in a 15% improvement in recommendation relevance based on user feedback.

        I am passionate about creating ethical, transparent, and user-centric AI systems that deliver meaningful value. Outside of work, I enjoy staying updated with the latest advancements in AI/ML, contributing to open-source projects like Hugging Face Transformers, and participating in technical communities such as Kaggle and Meetup groups. I am always open to connecting with professionals who share a passion for innovation and problem-solving. Let’s connect and explore how we can collaborate to build the future of intelligent systems!''',
                    'preferred_work_types': ['FULL_TIME', 'INTERNSHIP'],
                    'experience_level': 'Entry level',
                    'expected_salary': None,
                    'skills': ['python', 'scikitlearn', 'tensorflow', 'pandas']
                    }
        user2 = {
            'title': 'Graphics Designer',
            'about': '''Creative Graphics Designer with 5 years of experience in crafting visually compelling designs for digital and print media. Skilled in Adobe Creative Suite, including Photoshop, Illustrator, and InDesign, with a strong focus on branding, marketing materials, and user interface design. Passionate about translating client visions into impactful visual stories that resonate with target audiences.

        In my current role, I have successfully led design projects for various industries, delivering high-quality work under tight deadlines. My expertise includes logo design, social media graphics, and website mockups, ensuring consistency in brand identity across all platforms. I am also proficient in motion graphics and video editing, using tools like After Effects and Premiere Pro to create engaging multimedia content.

        I thrive in collaborative environments, working closely with marketing teams, developers, and clients to achieve project goals. My attention to detail and commitment to excellence have earned me recognition for delivering designs that exceed expectations. Outside of work, I enjoy exploring new design trends, participating in design challenges, and mentoring aspiring designers.''',
            'preferred_work_types': ['FULL_TIME', 'PART_TIME'],
            'experience_level': 'Mid level',
            'expected_salary': None,
            'skills': ['photoshop', 'illustrator', 'indesign', 'aftereffects']
        }

        user3 = {
            'title': 'Digital Marketer',
            'about': '''Results-driven Digital Marketer with 6 years of experience in developing and executing data-driven marketing strategies to boost brand awareness and drive customer engagement. Proficient in SEO, SEM, social media marketing, email campaigns, and analytics tools like Google Analytics and HubSpot. Adept at creating targeted campaigns that deliver measurable ROI.

        In my current role, I have successfully managed multi-channel marketing campaigns, increasing website traffic by 40% and improving conversion rates by 25%. My expertise includes content marketing, pay-per-click advertising, and influencer collaborations, ensuring alignment with business objectives. I am also skilled in A/B testing and performance analysis to optimize campaign effectiveness.

        I am passionate about staying ahead of digital marketing trends and leveraging innovative techniques to connect with audiences. Outside of work, I enjoy attending marketing conferences, contributing to industry blogs, and networking with professionals in the field.''',
            'preferred_work_types': ['PART_TIME', 'CONTRACT'],
            'experience_level': 'Mid level',
            'expected_salary': None,
            'skills': ['seo', 'sem', 'googleanalytics', 'hubspot']
        }

        users = [user1, user2, user3]
        for i, usr_data in enumerate(users):
            usr = user(
                user_id= i + 1,
                title=usr_data['title'],
                about=usr_data['about'],
                preferred_work_types=usr_data['preferred_work_types'],
                experience_level=usr_data['experience_level'],
                expected_salary=usr_data['expected_salary'],
                skills=usr_data['skills']
            )
            
            users[i] = usr

            print(f"Embedding user {usr.user_id} ...")
            embeddings = user_embed.embed(usr.model_copy())
            print(f"Storing user {usr.user_id} embeddings ...")
            db.store_user_embeddings(usr.user_id, embeddings)
            print(f"User {usr.user_id} embeddings stored successfully")
        
        st.session_state.users = users


st.title("Job Recommender System")
st.write("This is a job recommender system that recommends jobs based on user profiles.")
st.divider()
job_objs = st.session_state.jobs
users = st.session_state.users
weight = st.session_state.weights

with st.sidebar:
    select_user = st.selectbox("Select a user", options=[f"User {usr.user_id}: {usr.title}" for usr in users] + ["Custom User"])
    
    if select_user == "Custom User":
        title = st.text_input("Title")
        about = st.text_area("About")
        preferred_work_types = st.multiselect("Preferred Work Types", options=['CONTRACT', 'FULL_TIME', 'INTERNSHIP', 'PART_TIME', 'TEMPORARY', 'VOLUNTEER'])
        
        usr = user(
            user_id=0,
            title= title,
            about= about,
            preferred_work_types= preferred_work_types,
            experience_level= None,
            expected_salary=None,
            skills= None
        )
    else:
        usr = users[int(select_user[5]) - 1]
        st.header("User Information")
        st.markdown(f"**Title:**")
        st.text(f"{usr.title}")
        st.markdown(f"**About:**")
        st.text(f"{usr.about}")
        st.markdown(f"**Preferred Work Types:** {', '.join(usr.preferred_work_types)}")
        st.markdown("---")
    
    button_recommend = st.button("Recommend Jobs", use_container_width=True)
    
    st.markdown("---")
    st.header("Specify Recommender Weights")
    
    title = st.slider("Title", min_value=0.0, max_value=1.0, step=0.1, value=weight.title)
    content = st.slider("About", min_value=0.0, max_value=1.0, step=0.1, value=weight.content)
    work_type = st.slider("Work Type", min_value=0.0, max_value=1.0, step=0.1, value=weight.work_type)
    st.markdown(f"**Summation of weights:** {title + content + work_type}")
    
    set_weights = st.button("Set Weights", use_container_width=True)
    if set_weights:
        weight.title = title
        weight.content = content
        weight.work_type = work_type
        
        if weight.title + weight.content + weight.work_type != 1.0:
            st.warning("The sum of weights should equal 1.0")
        else:
            st.session_state.weights = weight


    if button_recommend:
        if select_user == "Custom User":
            print(f"Embedding user {usr.user_id} ...")
            embeddings = user_embed.embed(usr)
            print(f"Storing user {usr.user_id} embeddings ...")
            db.store_user_embeddings(usr.user_id, embeddings)
            print(f"User {usr.user_id} embeddings stored successfully")

        recommendations = recommender.user_job_recommend(
            user_id= usr.user_id,
            jobs_ids= [jb.job_id for jb in job_objs],
            recommender_weights= st.session_state.weights,
        )
        st.session_state.recommendations = filter_recommendations(recommendations, max_recommendations=10)


if st.session_state.recommendations:
    st.header("Recommended Jobs")
    for job_id in st.session_state.recommendations:
        job = job_objs[job_id]
        with st.container():
            st.markdown(f"### {job.title}")
            st.markdown(f"**Work Type:** {job.work_type}")
            st.markdown(f"**Description:** {job.content[:200]}...")
            if st.button(f"View More", key=f"view_more_{job_id}"):
                st.markdown(f"**Full Description:** {job.content}")
            st.markdown("---")