import openai
from googletrans import Translator
t=Translator()

openai.api_key = "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"

def chat(message):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=message,
        temperature=0.7,
        max_tokens=50,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response.choices[0].text.strip()

def main():
    print("ChatBot is ready. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = chat(user_input)
        response=t.translate(response,dest="en")
        print(response.src)
        print("ChatBot:", response.text)
main()
