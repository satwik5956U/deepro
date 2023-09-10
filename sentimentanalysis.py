import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertTokenizer, BertForSequenceClassification

# Load the GPT-2 model and tokenizer for text generation
gpt2_model_name = "gpt2"
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)

# Load the BERT model and tokenizer for sentiment analysis
bert_model_name = "bert-base-uncased"
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_model = BertForSequenceClassification.from_pretrained(bert_model_name)

def analyze_sentiment(text):
    # Tokenize the input text for sentiment analysis
    input_ids = bert_tokenizer.encode(text, add_special_tokens=True)
    input_ids = torch.tensor(input_ids).unsqueeze(0)

    # Perform sentiment analysis
    with torch.no_grad():
        outputs = bert_model(input_ids)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)[0].tolist()
    label = "positive" if probabilities[1] > probabilities[0] else "negative"

    return label

def chat_with_sentiment_analysis():
    print("Chat with the chatbot (type 'exit' to end)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        
        # Analyze sentiment of user input
        sentiment = analyze_sentiment(user_input)

        # Generate a response using GPT-2 based on sentiment
        bot_input = f"You ({sentiment} sentiment): {user_input}"
        input_ids = gpt2_tokenizer.encode(bot_input, return_tensors="pt")
        bot_response = gpt2_model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
        
        bot_response_text = gpt2_tokenizer.decode(bot_response[0], skip_special_tokens=True)
        print(f"Chatbot: {bot_response_text}")

if __name__ == "__main__":
    chat_with_sentiment_analysis()

