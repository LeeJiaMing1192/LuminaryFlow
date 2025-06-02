from self_deter_construct import ModelHandler_AutoGen
import torch

# Initialize the handler with Llama 3 8B
handler = ModelHandler_AutoGen(
    api_key="",
    time_link=True

    # deep_reasoning=False
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
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    response = handler.classify_and_respond(
        prompt, 
        sparse_activation=sparse_activation,
        unify=unify
    )


    print(response.get("type"))
    

# Test cases
test_prompts = [ 
    ("Cany you remind me of my birth day please.... its on March 22nd", False, False), 

]

for prompt, sparse, unify in test_prompts:
    run_test(prompt, sparse_activation=sparse, unify=unify)

