from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

def compute_resume_job_similarity(resume_text: str, job_text: str) -> float:
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    job_embedding = model.encode(job_text, convert_to_tensor=True)

    similarity_score = util.cos_sim(resume_embedding, job_embedding).item()
    return round(similarity_score, 3)
