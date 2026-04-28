from google import genai
import os

def test_gemini_3_1():
    project_id = "reviewops-493717"
    # Using the full resource ID found in the inventory scan
    model_id = "publishers/google/models/gemini-3.1-flash-lite-preview"
    location = "us-central1"
    
    print(f"--- Vertex AI Gemini 3.1 Test ---")
    print(f"Project: {project_id}")
    print(f"Model ID: {model_id}")
    print(f"Location: {location}")
    print(f"----------------------------------")

    try:
        client = genai.Client(
            vertexai=True, 
            project=project_id, 
            location=location
        )
        
        print(f"Sending prompt to {model_id}...")
        response = client.models.generate_content(
            model=model_id,
            contents="Say 'Hello! Gemini 3.1 Flash-Lite is online.' Keep it short."
        )
        
        print(f"\n[SUCCESS] Received Response:")
        print(f">>> {response.text.strip()}")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")

if __name__ == "__main__":
    test_gemini_3_1()
