import re
from collections import Counter
from google import genai
from PIL import ImageGrab
import os
from datetime import datetime
import json
###############################Utils section##########################################
import os
import subprocess
os.environ["PYTHONUTF8"] = "1"
import sys
import io
import psutil

from datetime import datetime # Corrected import: import the datetime CLASS directly
# from Acessing_app.youtube_play import search_youtube_and_play
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

client = genai.Client(api_key="AIzaSyABw1ZSfEHxfdKCHybrDNHmNV2BjSuhMyM")

def capture_and_save_screenshot(filename_prefix="screenshot"):
    """
    Captures the current screen screenshot and saves it in the root folder
    of the project with a timestamp in the filename.

    Args:
        filename_prefix (str, optional): The prefix for the filename.
                                         Defaults to "screenshot".
    """
    try:
        # Capture the entire screen
        screenshot = ImageGrab.grab()

        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Construct the filename with prefix and timestamp and .jpg extension
        filename = f"temp_shot.jpg"

        # Get the root directory of the project (where this script is likely located)
        root_dir = os.getcwd()
        filepath = os.path.join(root_dir, filename)

        # Save the screenshot as a JPG file
        screenshot = screenshot.convert("RGB")  # Convert to RGB to avoid JPEG issues with alpha channel
        screenshot.save(filepath, "JPEG")

        print(f"Screenshot saved")

    except Exception as e:
        print(f"An error occurred while capturing or saving the screenshot: {e}")

import re
from collections import Counter
from google import genai
from PIL import ImageGrab
import os
from datetime import datetime
import json
import subprocess
import psutil
import sys
import io

class ModelHandler:
    def __init__(self, api_key , time_link):
        """Initialize with your Google Gemini API key"""
        self.client = genai.Client(api_key=api_key)
        self.expert_models = {}
        self.expert_descriptions = {}
        self.time_queing_mode = time_link
        
        print("[INFO] Gemini ModelHandler initialized")

    def extract_classification(self, text):
        """Extract all classification characters from response"""
        matches = re.findall(r'########\s*([A-Z])\s*########', text)
        return matches if matches else None

    def extract_answer(self, text):
        """Extract content between $ or $$ delimiters."""
        match = re.search(r'\$+\s*(.*?)\s*\$+', text, re.DOTALL)  # Matches 1+ $
        return match.group(1).strip() if match else text
    
    def extract_strings_in_quotes(self, text):
        """Extract all text between double quotes"""
        return re.findall(r'"(.*?)"', text)

    def run_external_script(self, search_term):
        """Run the external script asynchronously with subprocess, ensuring only one instance runs at a time"""
        # First stop any existing instance
        try:
            # Find and kill existing python processes running youtube_play.py
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if (proc.info['name'] in ('python.exe', 'pythonw.exe') and 
                        'youtube_play.py' in ' '.join(proc.info['cmdline'] or [])):
                        proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
        except Exception as e:
            print(f"Error terminating existing instances: {e}")

        # Then run the new instance
        if isinstance(search_term, list):
            search_term = ' '.join(search_term)  # Join list into a single string
        
        try:
            subprocess.Popen([
                sys.executable, 
                'C:\\Users\\ADMIN\\Desktop\\Vtuber_backend\\Acessing_app\\youtube_play.py', 
                search_term
            ])
        except Exception as e:
            print(f"Error starting YouTube player: {e}")


    def add_reminder_to_queue(self , reminder_text, initial_time_seconds):
        """
        Adds a new reminder with its initial queuing time to a JSON file.
        This function reads the entire file, appends the new reminder in memory,
        and then writes the updated list back to the file, ensuring valid JSON structure.

        Args:
            reminder_text (str): The description of the reminder.
            initial_time_seconds (int): The time in seconds until the reminder should trigger.
            filepath (str): The path to the JSON file.
        """

        filepath='reminders_queue.json'
        reminders = []

        # 1. Read existing data
        # Check if the file exists and has content to avoid JSONDecodeError on empty/non-existent file
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            try:
                with open(filepath, 'r') as f:
                    reminders = json.load(f)
                # Ensure the loaded data is a list; if not, re-initialize
                if not isinstance(reminders, list):
                    print(f"Warning: '{filepath}' contains non-list JSON. Re-initializing with empty queue.")
                    reminders = []
            except json.JSONDecodeError:
                print(f"Warning: '{filepath}' is corrupted or not valid JSON. Starting with an empty queue.")
                reminders = []
        else:
            print(f"'{filepath}' not found or is empty. A new file will be created.")

        # 2. Create the new reminder entry
        new_reminder_entry = {
            "reminder": reminder_text,
            "time_in_seconds": initial_time_seconds,
            "active": True # Default to active when added
        }

        # 3. Append the new entry to the in-memory list
        reminders.append(new_reminder_entry)

        # 4. Write the *entire updated list* back to the file, overwriting its previous content
        with open(filepath, 'w') as f:
            json.dump(reminders, f, indent=2) # indent=2 for pretty-printing

        print(f"Successfully added '{reminder_text}' with a queuing time of {initial_time_seconds} seconds to '{filepath}'.")
    def format_timestamp(self, iso_timestamp):
        """Formats ISO timestamp into a more readable format."""
        try:
            dt = datetime.fromisoformat(iso_timestamp)
            return dt.strftime("%b %d, %Y at %I:%M %p")
        except (ValueError, TypeError):
            return iso_timestamp

    def load_sender_message_pairs(self, data, max_pairs=5):
        """Creates a formatted conversation history with message context and timestamps."""
        conversation_history = []
        pair_count = 0
        
        for entry in reversed(data):
            if "You" not in entry:
                continue
                
            # Extract metadata
            message_type = entry.get("Message in", "Direct message")
            timestamp = self.format_timestamp(entry.get("Timestamp", ""))
            
            # Get bot response
            you_text = self.extract_text(entry["You"])
            if not you_text:
                continue
                
            # Find sender key (skip fixed metadata keys)
            sender_key = next(
                (key for key in entry 
                if key not in ["You", "Message in", "Timestamp"]),
                None
            )
            
            if sender_key:
                # Clean sender name and get message
                clean_sender = sender_key.split("//")[0].strip()
                sender_text = self.extract_text(entry[sender_key])
                if not sender_text:
                    continue
                
                # Format context prefix
                context_prefix = f"[{message_type}]"
                if timestamp:
                    context_prefix = f"[{message_type} - {timestamp}]"
                
                # Add to conversation
                conversation_history.append(f"{context_prefix} {clean_sender}: {sender_text}")
                conversation_history.append(f"You: {you_text}")
                pair_count += 1

            if pair_count >= max_pairs:
                break
        
        return "\n".join(conversation_history)

    def load_conversation_memory(self, file_path="conversation_memory_test.json", max_pairs=5):
        """Loads JSON data from file and extracts formatted conversation history with context."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return self.load_sender_message_pairs(data, max_pairs)
        except FileNotFoundError:
            print(f"Error: File not found at path: {file_path}")
            return ""
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from file: {file_path}")
            return ""

    def extract_text(self, data):
        """Helper method to extract text from various data formats"""
        if isinstance(data, str):
            try:
                loaded_data = json.loads(data)
                if isinstance(loaded_data, dict) and "prompt" in loaded_data:
                    return loaded_data["prompt"]
                elif isinstance(loaded_data, str):
                    return loaded_data
                else:
                    return None
            except json.JSONDecodeError:
                return data
        elif isinstance(data, dict) and "prompt" in data:
            return data["prompt"]
        elif isinstance(data, str):
            return data
        return None
    
    
    def parse_input_string(self , input_str):
    # First separate the header if it exists
        headers = []
        if input_str.startswith("########"):
            header_line = input_str.split('\n', 1)[0]
            headers = [h.strip() for h in header_line.split('########') if h.strip()]
            input_str = input_str.split('\n', 1)[-1].strip()
        
        # Split into individual sections first
        raw_sections = [s.strip() for s in input_str.split('$$$$$$$$') if s.strip()]
        
        # Process each section
        result = {}
        current_key = None
        current_values = []
        
        for section in raw_sections:
            if section in ['T', 'D', 'Y']:  # Add other markers as needed
                # Save previous section
                if current_key and current_values:
                    result[current_key] = current_values[0] if len(current_values) == 1 else current_values
                
                # Start new section
                current_key = section
                current_values = []
            else:
                # Add to current values
                try:
                    current_values.append(int(section))
                except ValueError:
                    current_values.append(section)
        
        # Save the last section
        if current_key and current_values:
            result[current_key] = current_values[0] if len(current_values) == 1 else current_values
        
        return result
    def get_current_human_readable_time(self):
        """
        Returns the current date and time in a simple, human-readable string format.
        Example: "Sunday, May 25, 2025, 11:56 PM" (or AM)
        """
        current_dt = datetime.now()

        # Format codes used:
        # %A: Full weekday name (e.g., "Sunday")
        # %B: Full month name (e.g., "May")
        # %d: Day of the month as a zero-padded decimal (e.g., "25")
        # %Y: Year with century (e.g., "2025")
        # %I: Hour (12-hour clock) as a zero-padded decimal (e.g., "11")
        # %M: Minute as a zero-padded decimal (e.g., "56")
        # %p: AM or PM
        human_readable_string = current_dt.strftime("%A, %B %d, %Y, %I:%M %p")

        return human_readable_string

    def classify_and_respond(self, user_prompt, sparse_activation=False, unify=True):
        """Main method to handle user prompts with multi-model processing
        
        Args:
            user_prompt (str): The user's input prompt
            sparse_activation (bool): If True, returns all activated models' responses
            unify (bool): If True and sparse_activation is True, combines all responses into one
                        If False, returns responses separately
        """


        current_time = self.get_current_human_readable_time()
        try:
            # Load conversation memory
            memory_string = self.load_conversation_memory()
            print(memory_string.strip())

            # Build expert options string
            expert_options = "\n".join(
                f"{char}) {desc}" 
                for char, desc in self.expert_descriptions.items()
            )


            time_linked = ""
            time_reminder_message_PROMPT = ""

            if (self.time_queing_mode == True):
                time_linked = "T) If the user tasks need to create schedule or an event reminder"
                time_reminder_message_PROMPT = "this is the complex part if its create a schedule or (T), you will give it like T$$$$$$$$ IN HERE SHOW THE REMINDER TAG FOR EXAMPLE'Good night' babe or 'Mamas birth day'  $$$$$$$$ DO NOT SHOW THIS BUT calculate the estimated countdown time in seconds put in here ONLY THE NUMBER ENDS WITH $$$$$$$$"


            classification_prompt = f"""
            HERE is the current time please remember that : {current_time}
            You sees the prompt and allow to choose:
            Available experts:
            M) Math problems (calculations, equations)
            D) Chat response (general conversation, VTuber style)
            Y) YouTube requests (play songs or videos)
            {time_linked}
            {expert_options}

            Here is the message: "{user_prompt}"

            Respond ONLY with: ########X######## where X is the classification character. 
            For complex tasks requiring multiple experts, provide multiple classifications 
            like ########X######## ########Y######## etc. ensure there is always D in teh classification
            then a NEW LINE
            After the classification you should put your message between $$$$ Message in here $$$$  but please remember that in one go there are lots of experts needed to be used so for fufilling lots of experts(when its needed) you give it follow this structure "expert repesentitve letter"$$$$$$$ message $$$$$$$$ so heres what it will look like when multi expert use:

            Y$$$$$$$$ Videos of cute cats$$$$$$$$$$$$
            {time_reminder_message_PROMPT}
            NEWLINE NEWLINE please
            ALWAYS have this:

            D$$$$$$$$ Your response to user chat or your motivation with cuteness based on what you saw $$$$$$$$
            """

            # Get Gemini's classification
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-preview-04-17",
                contents=classification_prompt
            )
            
            response_text = response.text if hasattr(response, 'text') else ""
            print(f"[RAW RESPONSE]\n{response_text}")

            # Process classification - now gets a list
            classifications = self.extract_classification(response_text)

            if classifications:
                print(f"[CLASSIFICATIONS FOUND] {', '.join(classifications)}")
                
                current_prompt = user_prompt
                accumulated_responses = []
                response_futures = []
                
                for classification in classifications:
                    try:
                        print(f"[PROCESSING MODEL {classification}]")
                        extract_for_time_dict = self.parse_input_string(response_text) ####### Parse into a dictionary

                        # print(extract_for_time_dict)
                        if classification == 'M':
                            result = self.solve_math(current_prompt)
                        elif classification == 'D':
                            chat =  extract_for_time_dict['D']

                            print(extract_for_time_dict['D'])
                            result = chat
                        elif classification == 'Y':
                            search_terms = self.extract_strings_in_quotes(current_prompt)
                            if search_terms:
                                self.run_external_script(search_terms)
                            result = f"Playing: {', '.join(search_terms)}"

                        elif classification == "T":
                            reminder = extract_for_time_dict['T'][0]  # Returns 'bedtime reminder'
                            time_value = extract_for_time_dict['T'][1]  # Returns 81000

                            print(reminder)

                            print(time_value)
                            self.add_reminder_to_queue(reminder, int(time_value)) # 2 hours from now
                           
                        elif classification in self.expert_models:
                            result = self.expert_models[classification](current_prompt)
                        else:
                            result = self.extract_answer(response_text)
                        
                        if result:
                            accumulated_responses.append({
                                "model": classification,
                                "response": str(result)
                            })
                            current_prompt = f"Previous output: {result}\nOriginal prompt: {user_prompt}"
                            
                    except Exception as e:
                        print(f"[MODEL {classification} ERROR] {str(e)}")
                        accumulated_responses.append({
                            "model": classification,
                            "error": str(e)
                        })
                
                if sparse_activation:
                    if unify:
                        # Combine all responses into a single string
                        combined_responses = "\n\n".join(
                            f"Expert model {resp['model']} response:\n{resp.get('response', resp.get('error', 'No response'))}"
                            for resp in accumulated_responses
                        )
                        
                        # Pass to a unifying model
                        unified_response = self.client.models.generate_content(
                            model="gemini-2.0-flash",
                            contents=f"Combine these expert responses into one coherent answer:\n\n{combined_responses}"
                        )
                        
                        return {
                            "type": "unified-sparse",
                            "models_used": [c for c in classifications],
                            "response": unified_response.text if hasattr(unified_response, 'text') else "No unified response"
                        }
                    else:
                        # Return all responses separately
                        return {
                            "type": "sparse-separate",
                            "models_used": [c for c in classifications],
                            "responses": accumulated_responses
                        }
                else:
                    # Original behavior - just return the accumulated responses
                    return {
                        "type": "multi-model",
                        "models_used": [c for c in classifications],
                        "responses": accumulated_responses
                    }
                    
            else:
                print("[NO CLASSIFICATION FOUND]")
                answer = self.extract_answer(response_text) if response_text else "No response generated"
                return {
                    "type": "default",
                    "response": answer
                }

        except Exception as e:
            print(f"[CLASSIFY AND RESPOND ERROR] {str(e)}")
            return {
                "type": "error",
                "response": f"System error: {str(e)}"
            }

    def add_expert(self, char, description, handler_function):
        """Add an expert handler"""
        self.expert_models[char] = handler_function
        self.expert_descriptions[char] = description
        print(f"[ADDED EXPERT] {char} - {description}")

    def solve_math(self, prompt):
        """Built-in math solver"""
        math_response = self.client.models.generate_content(
            model="gemini-1.5-flash",
            contents=f"Solve this math problem step-by-step: {prompt}"
        )
        solution = math_response.text if hasattr(math_response, 'text') else "No solution found"
        print(f"\n[MATH SOLUTION]\n{solution}")
        return solution

    def capture_and_save_screenshot(self, filename_prefix="screenshot"):
        """
        Captures the current screen screenshot and saves it in the root folder
        of the project with a timestamp in the filename.
        """
        try:
            # Capture the entire screen
            screenshot = ImageGrab.grab()

            # Create a timestamp for the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Construct the filename with prefix and timestamp and .jpg extension
            filename = f"temp_shot.jpg"

            # Get the root directory of the project
            root_dir = os.getcwd()
            filepath = os.path.join(root_dir, filename)

            # Save the screenshot as a JPG file
            screenshot = screenshot.convert("RGB")
            screenshot.save(filepath, "JPEG")

            print(f"Screenshot saved as {filepath}")

        except Exception as e:
            print(f"An error occurred while capturing or saving the screenshot: {e}")