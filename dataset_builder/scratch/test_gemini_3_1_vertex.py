import vertexai
from vertexai.generative_models import GenerativeModel
import google.auth

def test_gemini_3_1_vertex_sdk():
    project_id = "reviewops-493717"
    location = "us-central1"
    model_id = "gemini-3.1-flash-lite-preview"
    
    print(f"--- Vertex AI SDK (Legacy) Gemini 3.1 Test ---")
    
    try:
        vertexai.init(project=project_id, location=location)
        
        print(f"Initializing model: {model_id}...")
        model = GenerativeModel(model_id)
        
        print("Generating content...")
        response = model.generate_content("Say 'Hello! Gemini 3.1 via Vertex SDK is working.'")
        
        print(f"\n[SUCCESS] Received Response:")
        print(f">>> {response.text.strip()}")
        
    except Exception as e:
        print(f"\n[ERROR] {e}")

if __name__ == "__main__":
    test_gemini_3_1_vertex_sdk()
