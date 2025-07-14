import requests
import numpy as np
import faiss
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline

def data_website(url):
    data = requests.get(url).text
    # print("Data:", data)
    soup = BeautifulSoup(data, "html.parser")
    # print("Soup:", soup)
    text = [p.get_text(strip=True) for p in soup.find_all(["p", "div"])]
    # print("Text:", " ".join(text))
    
    return " ".join(text)

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # print("Splitter:", splitter.split_text(text))
    
    return splitter.split_text(text)

def embed_text(split):
    model_embed = SentenceTransformer("all-MiniLM-L6-v2")
    # print("Model_Embed:", model_embed)
    vectors = model_embed.encode(split)
    # print("Vectors:", vectors)
    
    return vectors, model_embed

def indexing_text(vectors):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    # print("Index:", index)

    return index

def similar_text(query, split, embed_model, index, top_k=2):
    query_vec = embed_model.encode([query])
    D, I = index.search(np.array(query_vec), top_k)
    
    return [split[i] for i in I[0]]

def model_llm():
    model_llm = "tiiuae/falcon-rw-1b"
    tokenizer = AutoTokenizer.from_pretrained(model_llm)
    model = AutoModelForCausalLM.from_pretrained(model_llm)
    pipe = pipeline(
        "text-generation",
        model=model, 
        tokenizer=tokenizer, 
        max_new_tokens=512, 
        pad_token_id=tokenizer.eos_token_id
        )
    
    return HuggingFacePipeline(pipeline=pipe)

def generate_answer(llm, context_split, question):
    context = "\n".join(context_split)
    prompt = f"""
    This is a intelligent AI that can answer the questions below. If it can't answer, say "I don't know".

    Context:
    {context}
    
    Question:
    {question}

    Answer:
    """
 
    return llm.invoke(prompt)

def main():
    print("---กำลังเตรียมข้อมูล---")
    
    url = "https://www.tilda.com/faqs/"
    text = data_website(url)
    split = split_text(text)
    vectors, embed_model = embed_text(split)
    index = indexing_text(vectors)
    llm = model_llm()
    
    while True:
        question = input("\n Qustion?: ")
        if question.lower() in ["exit", "e"]:
            break
        
        print("---กำลังเตรียมคำตอบ---")
        
        context_split = similar_text(question, split, embed_model, index)
        answer = generate_answer(llm, context_split, question)
        print(str(answer))

if __name__ == "__main__":
    main()