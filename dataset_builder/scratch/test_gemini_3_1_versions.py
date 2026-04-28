from google import genai
import os

def test_gemini_3_1_versions():
    project_id = "reviewops-493717"
    location = "us-central1"
    
    # Try versions and suffixes
    model_variations = [
        "gemini-3.1-flash-lite-preview",
        "gemini-3.1-flash-lite-preview@001",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-preview"
    ]
    
    print(f"--- Gemini 3.1 Version Hunt ---")
    
    client = genai.Client(
        vertexai=True, 
        project=project_id, 
        location=location
    )

    for model in model_variations:
        try:
            print(f"Testing {model}...", end=" ", flush=True)
            response = client.models.generate_content(
                model=model,
                contents="Hi"
            )
            print(f"\n[SUCCESS] {model} is working!")
            print(f">>> {response.text.strip()}")
            return
        except Exception as e:
            if "404" in str(e):
                print("404")
            else:
                print(f"Error: {str(e)[:50]}...")

if __name__ == "__main__":
    test_gemini_3_1_versions()
