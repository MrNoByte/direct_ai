from transformers import pipeline,  AutoTokenizer, AutoModelForSequenceClassification 

# name of pre-trained model on hugging face
model_name = "LondonStory/txlm-roberta-hindi-sentiment"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model = model, tokenizer = tokenizer)

res = classifier(["मुझे लगता है कि मैं तुमसे प्यार करता हूँ लेकिन असल में मैं तुमसे नफरत करता हूँ", "मुझे तुमसे प्यार है", "मुझे आप से नफरत करनी चाहिए"])
print(res)
