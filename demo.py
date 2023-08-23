import time

import gradio as gr
import torch
from transformers import AutoTokenizer
import yaml


class ModelHandler:
    def __init__(self, config):
        self.config = config
        self.loaded_model = None
        self.loaded_model_name = None
        self.tokenizer = None

    def load_model(self, model_name):
        if self.loaded_model_name != model_name:
            # Release the previous model if it's loaded
            self.release_model()
            # Load the new model
            self._load_model_from_name(model_name)

    def predict_spam(self, text, model_name):
        self.load_model(model_name)

        if self.loaded_model is None:
            raise ValueError("Model is not loaded")

        # Tokenize and process the input
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        )

        # Make predictions
        input_ids = inputs["input_ids"]
        attention_masks = inputs["attention_mask"]
        device = self.loaded_model.device

        t1 = time.time()
        with torch.no_grad():
            outputs = self.loaded_model(
                input_ids.to(device), attention_mask=attention_masks.to(device)
            )
        timediff = time.time() - t1
        timediff = "{:.4f}".format(timediff)

        # Extract prediction probabilities
        predicted_probabilities = outputs.logits.softmax(dim=1)[0]

        # Print the predicted probabilities for each class
        class_names = ["ham", "spam"]
        output = ""
        for class_name, probability in zip(class_names, predicted_probabilities):
            output += f"{class_name}: {probability:.4f}\n"
        output += f"Infer time: {timediff} sec"

        return output

    def release_model(self):
        if self.loaded_model is not None:
            # Release the model resources
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.loaded_model = None
            self.loaded_model_name = None
            self.tokenizer = None

    def _load_model_from_name(self, model_name):
        # Load tokenizer and model
        if model_name == "bert":
            pre_train = "textattack/bert-base-uncased-yelp-polarity"
            fine_tune = self.config.get("bert_finetune_path")
            from transformers import BertForSequenceClassification
            model = BertForSequenceClassification.from_pretrained(fine_tune)
        elif model_name == "distilbert":
            pre_train = "distilbert-base-uncased"
            fine_tune = self.config.get("distilbert_finetune_path")
            from transformers import DistilBertForSequenceClassification
            model = DistilBertForSequenceClassification.from_pretrained(fine_tune)
        else:
            raise ValueError("model_name is not supportted")
        # total_params = sum(p.numel() for p in model.parameters())
        # print(f"Number of parameters of {model_name}: {total_params}")
        
        # Init tokenizer
        tokenizer = AutoTokenizer.from_pretrained(pre_train)

        # Set model to evaluation mode
        if torch.cuda.is_available():
            torch.device("cuda:0")
            model.cuda()
        else:
            torch.device("cuda:0")
        model.eval()

        self.loaded_model = model
        self.loaded_model_name = model_name
        self.tokenizer = tokenizer


with open("./config-demo.yaml", "r") as f:
    config = yaml.safe_load(f)
model_handler = ModelHandler(config)

iface = gr.Interface(
    fn=model_handler.predict_spam,
    inputs=[
        gr.inputs.Textbox(label="Text"),
        gr.inputs.Dropdown(choices=["bert", "distilbert"], label="Select Model"),
    ],
    outputs=gr.outputs.Textbox(),
    title="Spam detection",
    description="Select a model and enter text to detect Spam.",
)

iface.launch(debug=True, share=True)
