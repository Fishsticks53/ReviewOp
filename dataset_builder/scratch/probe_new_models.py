from google import genai
import os

def test_new_models():
    project_id = "reviewops-493717"
    location = "us-central1"
    
    # Try 2.5 Flash Lite and 3.1 Pro to isolate the issue
    models_to_test = [
        "gemini-2.5-flash-lite",
        "gemini-3.1-pro-preview"
    ]
    
    print(f"--- Gemini New Model Probe ---")
    
    client = genai.Client(
        vertexai=True, 
        project=project_id, 
        location=location
    )

    for model in models_to_test:
        try:
            print(f"Testing {model}...", end=" ", flush=True)
            response = client.models.generate_content(
                model=model,
                contents="Hello"
            )
            print(f"\n[SUCCESS] {model} is active!")
            print(f">>> {response.text.strip()}")
        except Exception as e:
            if "404" in str(e):
                print("404 Not Found")
            else:
                print(f"Error: {e}")

if __name__ == "__main__":
    test_new_models()
