from transformers import AutoModelForCausalLM

model_name = "C:\\Users\\ADMIN\\Desktop\\Everything\\Llama3_8B_test\\Llama3_8B_test_model"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

print(model)