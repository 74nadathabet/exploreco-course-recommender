# ------------------ Imports & Setup -------------------
import os
import re
import json
import logging
import asyncio
import aiohttp
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from collections import OrderedDict

# ------------------ Env Variables -------------------
load_dotenv()
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")

# ------------------ App & Models -------------------
app = FastAPI()

class UserData(BaseModel):
    question: str
    user_answer: str

class TopicData(BaseModel):
    score: int
    questions: List[UserData]

class FullInput(BaseModel):
    topics: Dict[str, TopicData]
    total_time: int
    total_score: int
    user_job_role: str

# ------------------ Dataset & Models -------------------
df = pd.read_csv("RECommendationDSs.csv")
embedding_model = SentenceTransformer("fine_tuned_course_model")
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["Improved Description"])
df["embedding"] = df["Improved Description"].apply(lambda x: embedding_model.encode(x, convert_to_numpy=True))
embeddings_matrix = np.vstack(df["embedding"].values)

# ------------------ Fuzzy Setup -------------------
raw_score = ctrl.Antecedent(np.arange(0, 71, 1), 'raw_score')
total_score = ctrl.Antecedent(np.arange(0, 351, 1), 'total_score')
total_time = ctrl.Antecedent(np.arange(0, 3001, 1), 'total_time')
adjusted_score = ctrl.Consequent(np.arange(0, 71, 1), 'adjusted_score')

raw_score.automf(names=['low', 'medium', 'high'])
total_score.automf(names=['low', 'medium', 'high'])
total_time.automf(names=['fast', 'moderate', 'slow'])

adjusted_score['low'] = fuzz.trimf(adjusted_score.universe, [0, 0, 25])
adjusted_score['medium'] = fuzz.trimf(adjusted_score.universe, [20, 35, 50])
adjusted_score['high'] = fuzz.trimf(adjusted_score.universe, [45, 70, 70])

rules = [
    ctrl.Rule(raw_score['high'] & total_score['high'] & total_time['fast'], adjusted_score['high']),
    ctrl.Rule(raw_score['high'] & total_score['medium'] & total_time['moderate'], adjusted_score['medium']),
    ctrl.Rule(raw_score['medium'] & total_score['medium'] & total_time['moderate'], adjusted_score['medium']),
    ctrl.Rule(raw_score['low'] & total_score['low'], adjusted_score['low']),
    ctrl.Rule(total_time['fast'] & (raw_score['low'] | raw_score['medium']), adjusted_score['low']),
    ctrl.Rule(raw_score['high'] & total_score['low'] & total_time['slow'], adjusted_score['medium']),
    ctrl.Rule(raw_score['high'] & total_score['high'] & total_time['slow'], adjusted_score['medium']),
]

adjusted_ctrl = ctrl.ControlSystem(rules)
adjusted_sim = ctrl.ControlSystemSimulation(adjusted_ctrl)

# ------------------ Helpers -------------------
def calculate_wrong_questions(score, max_topic_score=70):
    missing = max_topic_score - score
    easy = medium = hard = 0
    while missing >= 50 and hard < 2:
        hard += 1; missing -= 50
    while missing >= 20 and medium < 2:
        medium += 1; missing -= 20
    while missing >= 5 and easy < 2:
        easy += 1; missing -= 5
    return easy, medium, hard


def compute_penalty(easy, medium, hard, total_score, total_time, score):
    if easy + medium + hard == 0: return 0
    penalty = easy*0.4 + medium*0.8 + hard*1.5
    if total_score >= 250: penalty *= 0.3
    elif total_score >= 200: penalty *= 0.5
    if total_time > 2000 and score < 25: penalty += 1
    return min(penalty, 5)

def apply_total_score_adjustment(score, total_score):
    if total_score >= 250: return min(score + 5, 70)
    elif total_score >= 200: return min(score + 3, 70)
    elif total_score < 150: return max(score - 3, 0)
    return score
def compute_fuzzy_adjusted_score(score, total_score_val, total_time_val):
    easy, medium, hard = calculate_wrong_questions(score)
    penalty = compute_penalty(easy, medium, hard, total_score_val, total_time_val, score)
    adjusted_sim.input['raw_score'] = score
    adjusted_sim.input['total_score'] = total_score_val
    adjusted_sim.input['total_time'] = total_time_val
    adjusted_sim.compute()
    fuzzy_output = adjusted_sim.output['adjusted_score']
    final_score = fuzzy_output - penalty
    return max(0, round(apply_total_score_adjustment(final_score, total_score_val), 2))

def determine_level(score):
    return "Advanced" if score >= 50 else "Intermediate" if score >= 25 else "Beginner"

def deduplicate_courses(courses):
    seen = OrderedDict()
    for course in courses:
        key = (course["Course Title"].strip(), course["URL"].strip())
        if key not in seen:
            seen[key] = course
    return list(seen.values())

# ------------------ GPT Prompt -------------------
def extract_json_from_response(text):
    try:
        return json.loads(re.search(r'\[\s*{.*?}\s*]', text, re.DOTALL).group(0))
    except:
        return []

async def call_gpt_with_topic_data(topic_blocks, user_job_role, session):
    prompt = f"""
You are an AI assistant specialized in recommending cybersecurity courses based on user performance.

User Job Role: {user_job_role}

The user completed a technical assessment. For each topic below, we provide:
- The topic name
- The user's estimated level: Beginner / Intermediate / Advanced
- A list of incorrectly answered questions, along with the user's wrong answer

Your task is to suggest exactly 2 highly relevant courses per topic that follow these guidelines:
- Must be directly related to the topic
- Should help the user improve cybersecurity skills relevant to their job role as a {user_job_role}
- Must be appropriate for the user's level
- Must address the user's misunderstandings
- Must be technical and practical

⚠️ Important: All recommended courses must be related to **cybersecurity**, even if the topic or job role is from a development field (e.g., Flutter, DevOps, etc). For example, suggest mobile app security for Flutter, not Flutter development itself.

Return ONLY valid JSON in the format:
```json
[
  {{
    "title": "<Course Title>",
    "description": "<How this course helps the user improve in this topic>"
  }}
]


Here are the user’s weak topics:
{topic_blocks}
"""


    payload = {
        "messages": [
            {"role": "system", "content": "You are a technical education GPT assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1200,
        "temperature": 0.7
    }

    url = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT_NAME}/chat/completions?api-version={AZURE_API_VERSION}"
    headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}

    async with session.post(url, headers=headers, json=payload) as resp:
        result = await resp.json()
        return extract_json_from_response(result["choices"][0]["message"]["content"])

# ------------------ Hybrid Recommender -------------------
def hybrid_recommend_by_description(description, alpha=0.5, beta=0.3, gamma=0.2):
    embed = embedding_model.encode(description, convert_to_numpy=True)
    tfidf_vec = vectorizer.transform([description])
    cos_sim = cosine_similarity([embed], embeddings_matrix)[0]
    tfidf_sim = cosine_similarity(tfidf_vec, tfidf_matrix)[0]
    keywords = df["Improved Description"].apply(lambda d: sum(1 for w in description.lower().split() if w in d.lower()) / len(description.split())).values
    final_score = alpha*cos_sim + beta*tfidf_sim + gamma*keywords
    df["Final_Score"] = final_score
    top = df.sort_values("Final_Score", ascending=False).drop_duplicates("Course Title").head(2)
    return top[["Course Title", "URL"]].to_dict(orient="records")

# ------------------ Main Endpoint -------------------
@app.post("/full-analysis/")
async def full_analysis(input_data: FullInput):
    fuzzy_results = {}
    weak_topics = []
    results_by_topic = {}

    async with aiohttp.ClientSession() as session:
        for topic_name, topic_data in input_data.topics.items():
            adj_score = compute_fuzzy_adjusted_score(
                topic_data.score,
                input_data.total_score,
                input_data.total_time
            )
            level = determine_level(adj_score)

            block = f"\n### Topic: {topic_name}\nLevel: {level}\n"
            for q in topic_data.questions:
                block += f"- Q: {q.question}\n  A: {q.user_answer}\n"
            topic_blocks = block + "\n"

            fuzzy_results[topic_name] = {
                "adjusted_score": adj_score,
                "level": level,
                "questions": [q.dict() for q in topic_data.questions]
            }

            # 🟢 تعديل هنا: بنبعت كل التوبيكس مش بس البيجينر
            weak_topics.append(topic_name)

            try:
                gpt_courses = await call_gpt_with_topic_data(
                    topic_blocks, input_data.user_job_role, session
                )
                if not gpt_courses:
                    continue

                hybrid_courses = []
                for course in gpt_courses:
                    hybrid_courses += hybrid_recommend_by_description(course["description"])

                unique_top_courses = deduplicate_courses(hybrid_courses)[:2]
                results_by_topic[topic_name] = unique_top_courses

            except Exception as e:
                logging.error(f"Error in topic {topic_name}: {str(e)}")
                results_by_topic[topic_name] = []

    return results_by_topic