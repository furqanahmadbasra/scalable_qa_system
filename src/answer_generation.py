import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client
# It will automatically look for the GROQ_API_KEY environment variable
client = Groq()

# Specify the model to use (updated to current supported model)
MODEL_NAME = "llama-3.1-8b-instant"

def construct_prompt(query, chunks):
    """
    Constructs the prompt for the LLM given the user query and retrieved chunks.
    """
    context = ""
    for idx, chunk in enumerate(chunks, 1):
        context += f"\n--- Excerpt {idx} (Source: {chunk['source']}, Page: {chunk['page']}) ---\n"
        context += chunk['text'] + "\n"

    prompt = f"""You are a helpful and accurate academic policy assistant for NUST (National University of Sciences and Technology).
Your task is to answer the student's question based ONLY on the provided handbook excerpts.

Context Excerpts:
{context}

Student Question: {query}

Instructions:
1. Answer the question directly using ONLY the information from the context excerpts above.
2. If the answer is not contained within the excerpts, you must state: "I cannot find this information in the provided handbook excerpts." Do not try to guess or use outside knowledge.
3. Keep your answer clear, concise, and professional.
4. Optionally, you can cite the excerpts you used (e.g., "According to Excerpt 1...").

Answer:"""
    return prompt

def generate_answer(query, chunks):
    """
    Calls the Groq API to generate an answer based on the query and retrieved chunks.
    Returns the generated text.
    """
    if not chunks:
        return "No relevant information found in the handbooks to answer your question."

    prompt = construct_prompt(query, chunks)

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=MODEL_NAME,
            temperature=0.1, # Low temperature for more factual, less creative responses
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating answer from Groq API: {e}"

if __name__ == "__main__":
    import sys
    
    # We only run the test if the API key is actually set
    if os.environ.get("GROQ_API_KEY"):
        print("Loading LSH index for actual retrieval test...")
        
        # Import the LSH retrieval functions
        from lsh_retrieval import load_lsh_index, hybrid_search
        
        # Load the index
        lsh_index, minhash_objects, simhash_fps, chunk_shingles, chunks = load_lsh_index()
        
        # Use a real query
        test_query = "What happens if a student fails a course?"
        
        print(f"\nQuerying: '{test_query}'")
        print("Retrieving top chunks via Hybrid LSH...")
        
        # Retrieve top 3 chunks
        retrieved_chunks = hybrid_search(
            test_query, 
            lsh_index, 
            minhash_objects, 
            simhash_fps, 
            chunk_shingles, 
            chunks, 
            top_k=3
        )
        
        print(f"Retrieved {len(retrieved_chunks)} chunks. Generating answer...")
        
        answer = generate_answer(test_query, retrieved_chunks)
        
        print("\n=== Generated Answer ===")
        print(answer)
        
        print("\n=== Excerpts Provided to the Model ===")
        for idx, chunk in enumerate(retrieved_chunks, 1):
            print(f"Excerpt {idx} (Source: {chunk['source']} p.{chunk['page']}, Score: {chunk['score']}):")
            print(chunk['text'][:200] + "...\n") # Print first 200 chars to avoid flooding terminal
    else:
        print("GROQ_API_KEY not set in environment or .env file. Skipping test.")
