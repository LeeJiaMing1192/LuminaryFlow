from model_constructer import ModelHandler_AutoGen
import torch
from together import Together
from openai import OpenAI
# Initialize the handler with Llama 3 8B


client = Together(api_key="")
client_openRouter = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="",
)

# def my_art_handler_function(prompt):
#     print("\n[DEBUG ART HANDLER] Art expert processing prompt...")
#     return {"status": "success", "response": "TESTING ART EXPERT"}

handler = ModelHandler_AutoGen(
    api_key="",
    time_link=True,
    google_search_using="auto"


)
def advanced_complex_multi_language(prompt):
    

    response = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    messages=[
      {
        "role": "user",
        "content": prompt
      }
    ]
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content



def korean_english_expert(prompt):
    response = client.chat.completions.create(
    model="lgai/exaone-3-5-32b-instruct",
    messages=[
      {
        "role": "user",
        "content": prompt
      }
    ]
    )
    print(response.choices[0].message.content)
    return(response.choices[0].message.content)



def coding_model(prompt):
    completion = client_openRouter.chat.completions.create(

   extra_body={},
   model="qwen/qwen-2.5-coder-32b-instruct:free",
   messages=[
    {
      "role": "user",
      "content": "What is the meaning of life?"
    }
    ]
    )
    return completion.choices[0].message.content



def indic_language_reasoning(prompt):
    
    completion = client_openRouter.chat.completions.create(
    
    extra_body={},
    model="sarvamai/sarvam-m:free",
    messages=[
        {
        "role": "user",
        "content": prompt
        }
    ]
    )
    return completion.choices[0].message.content




def software_engineering_expert(prompt):
    
    completion = client_openRouter.chat.completions.create(
    
    extra_body={},
    model="mistralai/devstral-small:free",
    messages=[
        {
        "role": "user",
        "content": prompt
        }
    ]
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content


def chinese_biligual(prompt):
    completion = client_openRouter.chat.completions.create(
  
    extra_body={},
    model="thudm/glm-4-32b:free",
    messages=[
        {
        "role": "user",
        "content": prompt
        }
    ]
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content


def japanese_bilingual(prompt):
    completion = client_openRouter.chat.completions.create(

    extra_body={},
    model="shisa-ai/shisa-v2-llama3.3-70b:free",
    messages=[
        {
        "role": "user",
        "content": prompt
        }
    ]
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content

def creative_writing_model(prompt):
    completion = client_openRouter.chat.completions.create(
  
    extra_body={},
    model="arliai/qwq-32b-arliai-rpr-v1:free",
    messages=[
        {
        "role": "user",
        "content": prompt
        }
    ]
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content



def mathematical_reasoning_model(prompt):
    
    completion = client.chat.completions.create(
    
    extra_body={},
    model="thudm/glm-z1-32b:free",
    messages=[
        {
        "role": "user",
        "content": prompt
        }
    ]
    )
    print(completion.choices[0].message.content)

handler.add_expert("H" , "You will choose this if the user prompt is a complex prompt requires " , advanced_complex_multi_language)
handler.add_expert("K" , "You will choose this if the user prompt is requires you to analyze tradiotion medicine or a heavily flooded prompt with forgein Korean" , korean_english_expert)
handler.add_expert("C" , "You will choose this if the user prompt is requires you to solve or generate complex probnlem involving coding", coding_model)
handler.add_expert("I" , "You will choose this if the user prompt is requires you to work with indic languages", indic_language_reasoning)
handler.add_expert("Z" , "You will choose this if the user prompt is requires you to work with software engineering code, this model is the top SWE bench performer", software_engineering_expert)
handler.add_expert("B" , "You will choose this if the user prompt is requires you to work with chinese this is the best chinese biligual model so far", chinese_biligual)
handler.add_expert("P" , "You will choose this if the user prompt is requires you to work with japanese heavy overall task", japanese_bilingual)
handler.add_expert("J" , "You will choose this if the user prompt is requires you to work with creative writing and role play tasks with the user", creative_writing_model)
def run_test(prompt, sparse_activation=False, unify=True):
    print("\n" + "="*50)
    print(f"\n[TEST START] Processing prompt: '{prompt}'")
    print(f"[MODE] Sparse Activation: {sparse_activation} | Unify: {unify}")

    
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
    ("Prompt: The Silk Scroll Mystery You're an AI historian-assistant hired by an international museum to authenticate and reconstruct a fragmented ancient silk scroll discovered in Kyoto. This scroll is believed to be: A diplomatic artifact exchanged during the Tang Dynasty (China) and Nara Period (Japan). Written in Classical Chinese, Old Japanese (kanbun), and Sanskrit-based Indic script (possibly Buddhist textual influence). Accompanied by images of the scroll fragments, marginalia with symbols, and ink sketches. 🧠 Your Multimodal Reasoning Challenge: > Your task is to analyze the scroll’s text, script styles, and imagery, cross-reference with historical timelines and treaties, and: > > 1. Determine the date and provenance of the scroll fragments. > 2. Identify intercultural references indicating political, religious, or trade-based significance. > 3. Build a translation flow explaining how each language layer contributes to the overall message. > 4. Suggest preservation and display strategies based on material culture knowledge. Output: A translated section (selecting 1–2 meaningful lines and explaining them). A timeline of cross-cultural exchanges inferred from the content. A visual annotated map showing the scroll’s likely journey across dynasties and regions.", False, False), 

]

for prompt, sparse, unify in test_prompts:
    run_test(prompt, sparse_activation=sparse, unify=unify)

