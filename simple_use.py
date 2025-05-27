# from LLM_constructure import ModelHandler
# from together import Together

# client = Together()


# handler = ModelHandler(api_key="AIzaSyABw1ZSfEHxfdKCHybrDNHmNV2BjSuhMyM")
# # Add custom experts if needed


# def my_art_handler_function(prompt):
#     print("Art expert activated")
#     print(prompt + " expert Art")


# def math_model(prompt):
#     print("Math expert activated")
#     print(prompt + " expert MATH")

# def literature_model(prompt):
#     print("Literature expert activated")
#     print(prompt + " expert LIT")


# def image_gen(prompt):
#     print("Image expert activated")
#     print(prompt + " expert Image")


# def code_expert(prompt):
#     response = client.chat.completions.create(
#     model="Qwen/Qwen2.5-Coder-32B-Instruct",
#     messages=[{"role": "user", "content": prompt}]
#     )
#     print(response.choices[0].message.content)

# handler.add_expert('A', "Art expert", my_art_handler_function)
# handler.add_expert('M', "Math expert", math_model)
# handler.add_expert('L', "Literature expert", literature_model)
# handler.add_expert('I', "Image generation", image_gen)

# # Prompt designed to potentially trigger both Art and Math experts
# prompt_art_math = "Analyze the use of the golden ratio in the composition of a famous painting."
# response_art_math = handler.classify_and_respond(prompt_art_math)
# print(f"\nResponse for Art and Math prompt: {response_art_math}")

# # Prompt designed to potentially trigger both Literature and Art experts
# prompt_lit_art = "Describe the visual imagery and symbolism in a poem by Emily Dickinson."
# response_lit_art = handler.classify_and_respond(prompt_lit_art)
# print(f"\nResponse for Literature and Art prompt: {response_lit_art}")

# # Prompt designed to potentially trigger both Math and Literature experts (less likely but possible depending on classification logic)
# prompt_math_lit = "Explain the mathematical patterns found in the structure of a sonnet."
# response_math_lit = handler.classify_and_respond(prompt_math_lit)
# print(f"\nResponse for Math and Literature prompt: {response_math_lit}")

# # Prompt designed to potentially trigger the Image generation expert
# prompt_image = "Generate a picture of a serene landscape at sunset."
# response_image = handler.classify_and_respond(prompt_image)
# print(f"\nResponse for Image generation prompt: {response_image}")

# # Prompt designed to potentially trigger multiple experts (depending on your classification logic)
# prompt_multi = "Create a surreal painting inspired by the themes and mathematical structures in 'Alice in Wonderland'."
# response_multi = handler.classify_and_respond(prompt_multi)
# print(f"\nResponse for Multi-expert prompt: {response_multi}")

from LLM_constructure import ModelHandler




handler = ModelHandler(api_key="AIzaSyABw1ZSfEHxfdKCHybrDNHmNV2BjSuhMyM")

# Add custom experts with debug prints
def my_art_handler_function(prompt):
    print("\n[DEBUG ART HANDLER] Art expert processing prompt...")
    return f"Art analysis: {prompt}"

def math_model(prompt):
    print("\n[DEBUG MATH HANDLER] Math expert processing prompt...")
    return f"Math analysis: {prompt}"

def literature_model(prompt):
    print("\n[DEBUG LIT HANDLER] Literature expert processing prompt...")
    return f"Literary analysis: {prompt}"

def image_gen(prompt):
    print("\n[DEBUG IMAGE HANDLER] Image expert processing prompt...")
    return f"Image generation prompt: {prompt}"



handler.add_expert('A', "Art expert", my_art_handler_function) ##  My thuat
handler.add_expert('M', "Math expert", math_model) ## Toan
handler.add_expert('L', "Literature expert", literature_model) ## Ngu van
handler.add_expert('I', "Image generation", image_gen) ## Tao hinh anh
# handler.add_expert('C', "Coding expert", code_expert) 

def run_test(prompt, sparse_activation=False, unify=True):
    print("\n" + "="*50)
    print(f"\n[TEST START] Processing prompt: '{prompt}'")
    print(f"[MODE] Sparse Activation: {sparse_activation} | Unify: {unify}")
    
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
        print(response['response'])
    elif response['type'] == 'sparse-separate':
        print("\nSEPARATE EXPERT RESPONSES:")
        for resp in response['responses']:
            print(f"\nExpert {resp['model']}:")
            print(resp['response'])
    else:
        print(response)
    
    print("\n[TEST END]")
    print("="*50 + "\n")

# Test cases
test_prompts = [
    ("Analyze the use of the golden ratio in the composition of a famous painting.", True, False), ###
    ("Describe the visual imagery and symbolism in a poem by Emily Dickinson.", True, True),
    ("Explain the mathematical patterns found in the structure of a sonnet.", False, False),
]

for prompt, sparse, unify in test_prompts:
    run_test(prompt, sparse_activation=sparse, unify=unify)