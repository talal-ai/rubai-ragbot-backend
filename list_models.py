"""List all available Gemini models for your API key"""
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json

load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("[ERROR] GEMINI_API_KEY not found in environment variables")
    exit(1)

genai.configure(api_key=GEMINI_API_KEY)

print("[INFO] Fetching available Gemini models...")
print("=" * 60)

try:
    # List all available models
    models = genai.list_models()
    
    # Filter for models that support generateContent
    generate_models = []
    for model in models:
        if 'generateContent' in model.supported_generation_methods:
            generate_models.append({
                'name': model.name,
                'display_name': model.display_name,
                'description': model.description,
                'input_token_limit': model.input_token_limit,
                'output_token_limit': model.output_token_limit,
            })
    
    print(f"\n[SUCCESS] Found {len(generate_models)} models that support generateContent:\n")
    
    for i, model in enumerate(generate_models, 1):
        print(f"{i}. {model['name']}")
        print(f"   Display Name: {model['display_name']}")
        if model['description']:
            print(f"   Description: {model['description']}")
        print(f"   Input Tokens: {model['input_token_limit']:,}")
        print(f"   Output Tokens: {model['output_token_limit']:,}")
        print()
    
    # Show recommended free tier models
    print("=" * 60)
    print("\n[INFO] Recommended Free Tier Models:")
    print("   - gemini-pro (most stable)")
    print("   - gemini-1.5-pro (if available)")
    print("   - gemini-1.5-flash (if available)")
    print()
    
    # Show model names in a format ready to use
    print("=" * 60)
    print("\n[INFO] Model names you can use in config.py:")
    for model in generate_models:
        # Extract just the model name (remove 'models/' prefix if present)
        model_name = model['name'].replace('models/', '')
        print(f"   GEMINI_MODEL = '{model_name}'")
    
except Exception as e:
    print(f"[ERROR] Error listing models: {e}")
    import traceback
    traceback.print_exc()

