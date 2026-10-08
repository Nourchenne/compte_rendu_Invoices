import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from scipy.io import savemat
from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b",
)

class QuizQuestion(BaseModel):
    question: str = Field(description="The quiz question text")
    options: list[str] = Field(description="Four possible answers")
    correct_answer: str = Field(description="The correct option, must match one of 'options'")
    difficulty: str = Field(description="One of: easy, medium, hard")

structured_llm = llm.with_structured_output(QuizQuestion)

quiz_template = ChatPromptTemplate.from_template(
    '''Generate one multiple-choice quiz question about: {topic}.
    Don't repeat the previously generated questions {history}
    '''
)

quiz_gen = quiz_template | structured_llm


@tool
def save_quiz_to_mat(questions: list[dict], filename: str = "quiz.mat") -> str:
    """Sauvegarde une liste de questions de quiz dans un fichier MATLAB (.mat).

    Chaque question est un dict avec les clés : question, options,
    correct_answer, difficulty. Renvoie un message de confirmation.
    """
    data = {"quiz": questions}
    savemat(filename, data)
    return f"{len(questions)} question(s) sauvegardée(s) dans {filename}"


def print_quiz(q: QuizQuestion):
    print(f"\n  Q: {q.question}")
    for i, opt in enumerate(q.options, start=1):
        print(f"     {i}. {opt}")
    print(f"  Answer: {q.correct_answer}  |  Difficulty: {q.difficulty}")


gen_questions = list()

while True:
    topic = input("\nInsert a topic (/bye pour quitter) : ")

    if topic.lower() == "/bye":
        if gen_questions:
            # on appelle le tool avec .invoke() et un dict d'arguments
            result = save_quiz_to_mat.invoke({"questions": gen_questions})
            print(f"  -> {result}")
        print("Bye !")
        break

    quiz = quiz_gen.invoke({'topic': topic, 'history': gen_questions})
    print_quiz(quiz)
    # on convertit le modèle Pydantic en dict pour le tool
    gen_questions.append(quiz.model_dump())