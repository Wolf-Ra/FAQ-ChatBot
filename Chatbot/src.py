import fitz  # PyMuPDF
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Load and extract text from PDF
def extract_faq_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# Step 2: Parse questions and answers
def parse_faq(text):
    faq_pairs = re.findall(r"Q: (.*?)\nA: (.*?)(?=\nQ:|\Z)", text, re.DOTALL)
    return [(q.strip(), a.strip()) for q, a in faq_pairs]

# Step 3: Chatbot response using cosine similarity
class FAQChatbot:
    def __init__(self, faq_data):
        self.questions = [q for q, _ in faq_data]
        self.answers = [a for _, a in faq_data]
        self.vectorizer = TfidfVectorizer()
        self.question_vectors = self.vectorizer.fit_transform(self.questions)

    def get_response(self, user_input):
        user_vector = self.vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vector, self.question_vectors)
        max_index = similarities.argmax()
        max_score = similarities[0, max_index]

        if max_score < 0.3:
            return "Sorry, I didn't understand that. Can you rephrase it?"
        return self.answers[max_index]

# Main Execution
if __name__ == "__main__":
    nltk.download("punkt")

    pdf_path = "sample_faq_dataset.pdf"

    # Extract and parse FAQ
    raw_text = extract_faq_from_pdf(pdf_path)
    faq_data = parse_faq(raw_text)

    # Initialize chatbot
    chatbot = FAQChatbot(faq_data)

    print("📘 Welcome to the FAQ Chatbot! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break
        response = chatbot.get_response(user_input)
        print(f"Chatbot: {response}")
