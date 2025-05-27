from LLM_constructure import ModelHandler
import torch

# Initialize the handler with Llama 3 8B
handler = ModelHandler(
    api_key="AIzaSyABw1ZSfEHxfdKCHybrDNHmNV2BjSuhMyM",
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
    
    print("\n[CLASSIFICATION DEBUG]")
    if 'models_used' in response:
        print(f"Activated models: {response['models_used']}")
        if len(response['models_used']) > 1:
            if sparse_activation:
                print("MODE: Sparse Activation (parallel processing)")
                if unify:
                    print("RESULT TYPE: Unified response from all experts")
                else:
                    print("RESULT TYPE: Separate expert responses")
            else:
                print("MODE: Chain of Experts (sequential processing)")
        else:
            print("MODE: Single expert processing")
    else:
        print("MODE: Default response (no specific expert)")
    
    print("\n[FINAL RESPONSE]")
    if response['type'] == 'unified-sparse':
        print("\nUNIFIED RESPONSE:")
        print(response.get('response', 'No response generated'))
    elif response['type'] == 'sparse-separate':
        print("\nSEPARATE EXPERT RESPONSES:")
        for resp in response.get('responses', []):
            print(f"\nExpert {resp.get('model', 'UNKNOWN')}:")
            print(resp.get('response', resp.get('error', 'No response available')))
    else:
        print(response.get('response', 'No response generated'))
    
    if torch.cuda.is_available():
        print(f"\n[GPU MEMORY] Allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB | "
              f"Reserved: {torch.cuda.memory_reserved()/1024**2:.2f}MB")
    
    print("\n[TEST END]")
    print("="*50 + "\n")

# Test cases
test_prompts = [ 
    ("Cany you remind me of my birth day please.... its on March 22nd", False, False), 

]

for prompt, sparse, unify in test_prompts:
    run_test(prompt, sparse_activation=sparse, unify=unify)

