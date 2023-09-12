import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification 


# # model_name = "model" # default model name
# model_name = "distilbert-base-uncased-finetuned-sst-2-english" # default model name

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# classifier = pipeline("sentiment-analysis", model = model, tokenizer = tokenizer, return_tensors = "pt")

# res = classifier("i love you")
# print("res")

# # save_dir = "model"
# # tokenizer.save_pretrained(save_dir)
# # model.save_pretrained(save_dir)

save_dir = "sentiment_analysis/model/english-sent"
model2 = AutoModelForSequenceClassification.from_pretrained(save_dir)
tokenizer2 = AutoTokenizer.from_pretrained(save_dir)

classifier = pipeline("sentiment-analysis", model = model2, tokenizer = tokenizer2)
res = classifier("I love you")

print(res)





