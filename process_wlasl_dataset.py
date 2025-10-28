import json
WLASL_JSON_PATH = "C:\\Gautam\\Projects\\sign_language_translator\\model_development\\WLASL_v0.3.json"
def get_all_sign_words(json_path):
    sign_words = []
    seen = set()
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        for entry in data:
            word = entry['gloss']
            if word not in seen:
                sign_words.append(word)
                seen.add(word)
    except FileNotFoundError:
        print(f"Error: {json_path} file not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON file at {json_path}.")
        return []
    return sign_words        
def main():
    all_words = get_all_sign_words(WLASL_JSON_PATH)
    TARGET_WORDS = all_words[:5]
    print("Target words: ", TARGET_WORDS)    
if __name__ == "__main__":
    main()