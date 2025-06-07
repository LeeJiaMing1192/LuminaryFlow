LuminaryFlow
🚀 Overview
LuminaryFlow is a cutting-edge framework designed to streamline the development of [briefly describe what the framework is for, e.g., "high-performance web applications," "data processing pipelines," "machine learning models," etc.]. It provides a robust, scalable, and intuitive architecture that empowers developers to build complex systems with ease and efficiency.

Our goal with LuminaryFlow is to simplify common development challenges, accelerate project timelines, and ensure maintainability through clean, modular design principles.

✨ Features
Supports up to 5 Variations: Provides flexibility to develop and manage multiple variations of your application or model effortlessly.

Easy Expert and Tool Attachment: Seamlessly integrate custom experts and tools to extend the framework's capabilities.

Advanced Reasoning and Activation: Features intelligent tools capable of advanced reasoning to determine sequential or sparse activation of multiple experts.

Self-Built Long-Term Memory (FAISS): Incorporates a robust long-term memory retrieval system powered by FAISS for efficient data recall.

Short-Term Memory: Includes a short-term memory component for immediate context and improved conversational flow or processing.

Time-Queueing Memory: Utilizes a time-queueing memory system to manage and retrieve temporal data effectively.

Modular Architecture: Enables highly organized and reusable code components, promoting scalability and maintainability.



Extensible Plugin System: Easily integrate custom functionalities or third-party libraries.


[Add more features as needed]

📦 Installation
To get started with LuminaryFlow, follow these simple steps:

Prerequisites
[List any prerequisites, e.g., "Python (v3.9 or higher)", "A specific database", etc.]

Example: Python >= 3.9

Using pip
pip install luminaryflow

From Source
Clone the repository:

git clone https://github.com/LeeJiaMing1192/LuminaryFlow.git
cd luminaryflow

Install dependencies:

pip install -r requirements.txt

[Add any specific build steps if necessary, e.g., "Run build script: python setup.py build"]

💡 Usage
Here's a quick example of how to use LuminaryFlow in your project:

Advanced Usage Example (Python - Expert Integration)
from self_deter_construct import ModelHandler_AutoGen
import torch

# Initialize the handler with Gemini 2.5 Flash
handler = ModelHandler_AutoGen(
    api_key="",
    time_link=True
)

# Add custom experts with debug prints
def my_art_handler_function(prompt):
    print("\n[DEBUG ART HANDLER] Art expert processing prompt...")
    return {"status": "success", "response": "TESTING ART EXPERT"}

def math_model(prompt):
    print("\n[DEBUG MATH HANDLER] Math expert processing prompt...")
    return {"status": "success", "response": "TESTING MATH EXPERT"}

def literature_model(prompt):
    print("\n[DEBUG LIT HANDLER] Literature expert processing prompt...")
    return {"status": "success", "response": "TESTING LIT EXPERT"}

def image_gen(prompt):
    print("\n[DEBUG IMAGE HANDLER] Image expert processing prompt...")
    return {"status": "success", "response": f"Image generation prompt prepared: {prompt}"}

# Register the experts
handler.add_expert('A', "You will choose this if the user asks you to evaluate an artwork", my_art_handler_function)
handler.add_expert('M', "This expert you will choose when the user wanted to deal with complex equations", math_model)
handler.add_expert('L', "This expert you will choose when the user wanted to deal with Literature questions", literature_model)
handler.add_expert('I', "This expert you will choose when the user wanted to Image generation for example: Generate a picture of", image_gen)

def run_test(prompt, sparse_activation=False, unify=True):
    print("\n" + "="*50)
    print(f"\n[TEST START] Processing prompt: '{prompt}'")
    print(f"[MODE] Sparse Activation: {sparse_activation} | Unify: {unify}")
    # print(f"[MODEL] Llama 3 8B | Deep Reasoning: {handler.deep_reasoning}")


# Test cases
test_prompts = [
    ("Cany you remind me of my birth day please.... its on March 22nd", False, False),
]

for prompt, sparse, unify in test_prompts:
    run_test(prompt, sparse_activation=sparse, unify=unify)

For more detailed examples and advanced configurations, please refer to the Documentation folder.

🤝 Contributing
We welcome contributions to LuminaryFlow! If you're interested in helping improve the framework, please follow these guidelines:

Fork the repository.

Create a new branch for your feature or bug fix: git checkout -b feature/your-feature-name or bugfix/fix-issue-name.

Make your changes and ensure they adhere to our coding standards.

Write comprehensive tests for your changes.

Commit your changes with a clear and concise commit message.

Push your branch to your forked repository.

Open a Pull Request to the main branch of the original repository.

Please read our Contribution Guidelines for more details.

📄 License
LuminaryFlow is open-source software licensed under the MIT License, Apache 2.0 License. See the LICENSE file for more details.
