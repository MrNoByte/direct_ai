from transformers import pipeline,  AutoTokenizer, AutoModelForSequenceClassification 

# name of pre-trained model on hugging face
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model = model, tokenizer = tokenizer)

res = classifier(["I think i love you but i hate you.", "I love you", "I should hate you"])
print(res)
