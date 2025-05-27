import traceback
from flask import Flask, request, jsonify
from LLM_constructure import ModelHandler
from google import genai
from PIL import ImageGrab
import os
import json
import re
from datetime import datetime

app = Flask(__name__)

# Initialize the handler with your API key
handler = ModelHandler(api_key="AIzaSyABw1ZSfEHxfdKCHybrDNHmNV2BjSuhMyM")

def extract_display_name(prompt):
    """Extract display name from messages like 'Here's a message from {name} |...'"""
    pattern = r"from\s+(.*?)\s*\|"
    match = re.search(pattern, prompt)
    if match:
        return match.group(1).strip()
    return "Him"  # Default if no match found

# --- JSON Memory Store ---
class JSONMemoryStore:
    def __init__(self, file_path="conversation_memory_test.json"):
        self.file_path = file_path
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w") as f:
                json.dump([], f)

    def save_message(self,sender_name ,  user_data, model_data, message_type):
        """ Save user and model messages with additional metadata """
        try:
            # Extract display name if present
            
            # Create the message entry with all required fields
            message_entry = {
                sender_name: json.dumps({"prompt": user_data}),
                "You": model_data,
                "Message in": message_type,
                "Timestamp": datetime.now().isoformat()
            }
            
            with open(self.file_path, 'r+') as file:
                # Load the existing data from the file
                data = json.load(file)
                
                # Append the new message entry
                data.append(message_entry)
                
                # Move the file pointer to the beginning of the file
                file.seek(0)
                
                # Write the updated data back to the file
                json.dump(data, file, indent=4)
        except Exception as e:
            print(f"Error saving message: {e}")

    def load(self):
        try:
            with open(self.file_path, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                else:
                    print("⚠️ Corrupted memory detected. Resetting...")
                    return []
        except Exception as e:
            print(f"⚠️ Load error: {e}. Resetting file.")
            return []

    def get_conversation(self):
        return self.load()

memory_store = JSONMemoryStore("conversation_memory_test.json")

def get_mem():
    file_path = "conversation_memory_test.json"

    with open(file_path, "r") as f:
        data = json.load(f)

    memory_string = ""
    pair_count = 0
    max_pairs = 5

    for entry in data:
        # Find the user message key (could be any key except the fixed ones)
        user_key = None
        fixed_keys = ["You", "Message in", "Timestamp"]
        for key in entry:
            if key not in fixed_keys:
                user_key = key
                break
        
        if user_key and "You" in entry:
            try:
                user_prompt = json.loads(entry[user_key]).get("prompt", entry[user_key])
            except:
                user_prompt = entry[user_key]
                
            memory_string += f'{user_key}: {user_prompt.strip()} You: {entry["You"].strip()} '
            pair_count += 1
            
            if pair_count == max_pairs:
                break

    return memory_string


def capture_and_save_screenshot(filename="temp_shot.jpg"):
    try:
        print("[VISION] Capturing screenshot...")
        screenshot = ImageGrab.grab()
        screenshot.save(filename, "JPEG")
        print(f"[VISION] Screenshot saved to {filename}")
        return True
    except Exception as e:
        print(f"[VISION ERROR] Screenshot failed: {e}")
        return False

def code_expert(prompt):
    print(f"[EXPERT - CODE] Prompt received: {prompt}")
    response = handler.client.models.generate_content(
        model="gemini-1.5-flash",
        contents=f"Write clean, efficient code for: {prompt}\nInclude comments."
    )
    print(f"[EXPERT - CODE] Response: {response.text}")
    memory_store.save_message(prompt, response.text)
    return {"type": "code_expert", "response": response.text}

def science_expert(prompt):
    print(f"[EXPERT - SCIENCE] Prompt received: {prompt}")
    response = handler.client.models.generate_content(
        model="gemini-1.5-flash",
        contents=f"Provide accurate scientific explanation for: {prompt}\nCite sources when possible."
    )
    print(f"[EXPERT - SCIENCE] Response: {response.text}")
    memory_store.save_message(prompt, response.text)
    return {"type": "science_expert", "response": response.text}

def vision_expert(prompt):
    print(f"[EXPERT - VISION] Prompt received: {prompt}")
    capture_and_save_screenshot()
    client = genai.Client(api_key="AIzaSyABw1ZSfEHxfdKCHybrDNHmNV2BjSuhMyM")

    try:
        myfile = client.files.upload(file="C:/Users/ADMIN/Desktop/Vtuber_backend/temp_shot.jpg")
        print("[VISION] Uploaded screenshot for analysis.")
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[myfile, "Can you take a look at the screenshot and tell me: " + prompt]
        )
        print("[VISION] Initial visual analysis response:")
        print(response.text)

        json_start = response.text.find('```json')
        if json_start == -1:
            json_start = response.text.find('```')
        json_end = response.text.find('```', json_start + 3 if json_start != -1 else 0)

        if json_start != -1 and json_end != -1:
            json_string = response.text[json_start + 5:json_end].strip()
            if json_string.startswith('json'):
                json_string = json_string[4:].strip()
            
            detections = json.loads(json_string)
            extracted_texts = [item.get("text") or item.get("label") or "(no text)" for item in detections]
            combined_text = " ".join(extracted_texts)

            print("[VISION] Combined extracted text:")
            print(combined_text)
            try:
                print(response.text.label)
                final_response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=f"""Based on what the vision model is seeing, can you tell me the content of the picture?
                Hehe, remember you are still my girlfriend who is tech-nerdy and supportive , its like you are looking at my screen with me, here are our past conversations : {get_mem()}
    Here is my original prompt: {prompt}

    Vision model findings: {response.text}

    Please analyze this information and provide a detailed response. Hehe please reply like a caring and nerdy girlfriend"""
                )
                print("[VISION] Final Gemini analysis:")
                print(final_response.text)
                memory_store.save_message(prompt, final_response.text)
                return final_response.text
            except:
                final_response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=f"""Based on what the vision model is seeing, can you tell me the content of the picture?
                    Hehe, remember you are still my girlfriend who is tech-nerdy and supportive , its like you are looking at my screen with me, here are our past conversations : {get_mem()}
    Here is my original prompt: {prompt}

    Vision model findings: {response.text}

    Please analyze this information and provide a detailed response. Hehe please reply like a caring and nerdy girlfriend"""
                )
                print("[VISION] Final Gemini analysis:")
                print(final_response.text)
                memory_store.save_message(prompt, final_response.text)
                return final_response.text
        else:
            print("[VISION] No JSON found. Using raw response.")
            memory_store.save_message(prompt, response.text)
            return response.text

    except Exception as e:
        print(f"[VISION ERROR] JSON parsing failed: {e}")
        traceback.print_exc()
        memory_store.save_message(prompt, response.text)
        return response.text

# Register experts
# handler.add_expert("C", "Programming/coding questions", code_expert)
# handler.add_expert("S", "Science/technical questions", science_expert)
handler.add_expert("V", "You use this when I ask you to look at my screen or when you have to for example: 'Can you take a look at my screen and guess what am i doing?', 'Can you look at this thing with meee'", vision_expert)


@app.route("/classify", methods=["POST"])
def classify_res():
    try:
        data = request.get_json()
        print(data)  # {'prompt': '{"prompt": "Luuu", "chat_user": "Jia Ming", "msg_from": "Direct message"}'}

        # Parse the string inside 'prompt'
        parsed_prompt = json.loads(data['prompt'])

        # Now you can access the fields
        print(parsed_prompt['prompt'])       # Luuu
        print(parsed_prompt['chat_user'])    # Jia Ming
        print(parsed_prompt['msg_from'])     # Direct message
        temp_prompt = data.get("prompt")
        # print(prompt_raw)
        prompt_raw = temp_prompt
        print("XXXXXXXXXXXXXXXXXXXXX")
        sender =  parsed_prompt['chat_user']
        type_send =  parsed_prompt['msg_from']
        message_type = parsed_prompt['msg_from']

        # Attempt to extract transcription if it's JSON-encoded
        try:
            transcription_data = json.loads(prompt_raw)
            prompt = transcription_data.get("transcription", parsed_prompt['prompt'])
        except:
            prompt = parsed_prompt['prompt']

        print(f"[CLASSIFY] Received prompt: {prompt}")

        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        memory = memory_store.get_conversation()
        print(f"[CLASSIFY] Loaded memory: {len(memory)} entries")

        results = handler.classify_and_respond(prompt)
        print(results)
    
        memory_store.save_message(sender , prompt, results, type_send)
        return jsonify({"response": results})

    except Exception as e:
        print("[CLASSIFY ERROR] Exception occurred:")
        traceback.print_exc()
        return jsonify({"type": "error", "response": str(e)}), 500

# [Rest of your existing routes remain the same]

if __name__ == "__main__":
    print("[SERVER] Starting Flask server on port 7000")
    app.run(host="0.0.0.0", port=6000, debug=True)