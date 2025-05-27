import os
import torch
import sys
import io
from transformers import AutoTokenizer, AutoModelForCausalLM
import bitsandbytes as bnb
import re
from collections import Counter
import time
import psutil
import logging
from PIL import ImageGrab
import json
from datetime import datetime

class ModelHandler:
    def __init__(self, model_path, deep_reasoning=False):
        self.model_path = model_path
        self.loaded_tokenizer = None
        self.loaded_model = None
        self.expert_models = {}
        self.expert_descriptions = {}
        self.expert_model_stack = []
        self.extracted_answer = None
        self.deep_reasoning = deep_reasoning
        # self.resource_monitor = ResourceMonitor()
        print("[INFO] Llama 3 ModelHandler initialized")

    def extract_classification(self, text):
        """Extract all classification characters from response"""
        matches = re.findall(r'########\s*([A-Z])\s*########', text)
        return matches if matches else None

    def extract_answer(self, text):
        """Extract content between $ or $$ delimiters."""
        match = re.search(r'\$+\s*(.*?)\s*\$+', text, re.DOTALL)
        return match.group(1).strip() if match else text
    
    def extract_strings_in_quotes(self, text):
        """Extract all text between double quotes"""
        return re.findall(r'"(.*?)"', text)

    def run_external_script(self, search_term):
        """Run the external script asynchronously with subprocess"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if (proc.info['name'] in ('python.exe', 'pythonw.exe') and 
                        'youtube_play.py' in ' '.join(proc.info['cmdline'] or [])):
                        proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
        except Exception as e:
            print(f"Error terminating existing instances: {e}")

        if isinstance(search_term, list):
            search_term = ' '.join(search_term)
        
        try:
            print("bruh")
        except Exception as e:
            print(f"Error starting YouTube player: {e}")

    def load_model(self):
        """Load the Llama 3 model with appropriate quantization"""
        if not self.loaded_model:
            start_time = time.time()
            print(f"[MODEL LOADING] Loading {'deep reasoning' if self.deep_reasoning else 'standard'} model...")
            
            self.loaded_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.loaded_tokenizer.pad_token is None:
                self.loaded_tokenizer.pad_token = self.loaded_tokenizer.eos_token
            
            quantization_config = {
                "load_in_8bit": self.deep_reasoning,
                "load_in_4bit": not self.deep_reasoning
            }
            
            self.loaded_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                **quantization_config
            )
            
            load_time = time.time() - start_time
            print(f"[MODEL LOADED] in {load_time:.2f}s (in {'8-bit' if self.deep_reasoning else '4-bit'})")

    def unload_model(self):
        """Unload model and clear GPU memory"""
        if self.loaded_model:
            del self.loaded_model
            del self.loaded_tokenizer
            self.loaded_model = None
            self.loaded_tokenizer = None
            torch.cuda.empty_cache()
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
    def generate_response(self, prompt, max_new_tokens=256, temperature=0.7):
        """Generate response from Llama 3 model"""
        start_time = time.time()
        self.load_model()
        
        inputs = self.loaded_tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.loaded_model.device)
        
        with torch.no_grad():
            outputs = self.loaded_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                top_k=40,
                early_stopping=True,
                repetition_penalty=1.2
            )
        
        response = self.loaded_tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()  # Remove input prompt from response
        
        # Log performance metrics
        # inference_time = time.time() - start_time
        # self.resource_monitor.monitor(prompt, response, inference_time)
        
        return response

    def classify_and_respond(self, user_prompt, sparse_activation=False, unify=True):
        """Main method to handle user prompts with multi-model processing"""
        try:
            # Load conversation memory
            memory_string = self.load_conversation_memory()
            print(f"[MEMORY CONTEXT]\n{memory_string.strip()}")
            
            # Build expert options string
            expert_options = "\n".join(
                f"{char}) {desc}" 
                for char, desc in self.expert_descriptions.items()
            )

            classification_prompt = f"""Classify the user prompt and respond with a single letter either M  {', '.join(self.expert_descriptions.keys())}

User Prompt: "{user_prompt}"
Classification: always start with '########' then : IN HERE PLEASE ONLY 1 CHARACTER ONLY{', '.join(self.expert_descriptions.keys())}, M then ENDS'########### to ensure that its the final answer repeat it 4 times for me
When the task need some expert, you should evaluate ONLY the needed ones for this task and response like #########x########## ###############Y################
then you will answer to the user prompt ALWASY START with '$$' then your chat then ALWAYS END with'$$ here are the options:

M) If the prompt requires mathematical calculation for complex maths (e.g., "What is 5 times 7?", "Solve for x: 2x + 3 = 9" , "evaluate integral").

{expert_options}""" 

            # Get Llama 3's classification
            start_time = time.time()
            response_text = self.generate_response(
                classification_prompt,
                max_new_tokens=1000 if self.deep_reasoning else 400,
                temperature=0.3  # Lower temp for more precise classification
            )
            
            print(f"[RAW RESPONSE]\n{response_text}")
            print(f"[CLASSIFICATION TIME] {time.time() - start_time:.2f}s")

            # Process classification
            classifications = self.extract_classification(response_text)
            answer = self.extract_answer(response_text)

            if classifications:
                print(f"[CLASSIFICATIONS FOUND] {', '.join(classifications)}")
                
                current_prompt = user_prompt
                accumulated_responses = []
                
                for classification in classifications:
                    try:
                        print(f"[PROCESSING MODEL {classification}]")
                        
                        if classification == 'M':
                            result = self.solve_math(current_prompt)
                        elif classification == 'D':
                            result = answer if answer else "I don't have a response for that."
                        elif classification == 'Y':
                            search_terms = self.extract_strings_in_quotes(current_prompt)
                            if search_terms:
                                self.run_external_script(search_terms)
                            result = f"Playing: {', '.join(search_terms)}"
                        elif classification in self.expert_models:
                            result = self.expert_models[classification](current_prompt)
                        else:
                            result = answer if answer else "No specific expert available."
                        
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
                        
                        # Pass to Llama 3 for unification
                        unified_response = self.generate_response(
                            f"Combine these expert responses into one coherent answer:\n\n{combined_responses}",
                            max_new_tokens=1000 if self.deep_reasoning else 500
                        )
                        
                        return {
                            "type": "unified-sparse",
                            "models_used": classifications,
                            "response": unified_response
                        }
                    else:
                        # Return all responses separately
                        return {
                            "type": "sparse-separate",
                            "models_used": classifications,
                            "responses": accumulated_responses
                        }
                else:
                    # Original chain of experts behavior
                    return {
                        "type": "multi-model",
                        "models_used": classifications,
                        "responses": accumulated_responses
                    }
                
            else:
                print("[NO CLASSIFICATION FOUND]")
                answer = answer if answer else "No response generated"
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
        finally:
            self.unload_model()


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