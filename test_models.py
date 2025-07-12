from together import Together

# Replace "YOUR_API_KEY_HERE" with your actual API key from Together.ai
# client = Together(api_key="edbc5bdb99f79ca051591d8bcc538926664620bc1f0d5e0992df2942a59858b1")

# response = client.chat.completions.create(
#     model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
#     messages=[
#       {
#         "role": "user",
#         "content": "What are some fun things to do in New York?"
#       }
#     ]
# )
# print(response.choices[0].message.content)


from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)

completion = client.chat.completions.create(
#   extra_headers={
#     "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
#     "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
#   },
  extra_body={},
  model="qwen/qwen-2.5-coder-32b-instruct:free",
  messages=[
    {
      "role": "user",
      "content": "What is the meaning of life?"
    }
  ]
)
print(completion.choices[0].message.content)