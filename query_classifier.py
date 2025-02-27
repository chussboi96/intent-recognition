import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.pipeline.textcat_multilabel import DEFAULT_MULTI_TEXTCAT_MODEL
from typing import List, Dict
import random

class QueryClassifier:
    def __init__(self, labels: List[str]):
        """Initialize the classifier with given labels"""
        self.labels = labels
        self.nlp = spacy.blank("en")
        self.threshold = 0.5  # Define threshold as class attribute
        
        # Add the textcat_multilabel pipe
        self.textcat = self.nlp.add_pipe(
            "textcat_multilabel",
            config={
                "model": DEFAULT_MULTI_TEXTCAT_MODEL
            }
        )
        
        # Add labels to the textcat
        for label in self.labels:
            self.textcat.add_label(label)
            
        self.train_examples = []
        
    def add_training_examples(self, examples: List[Dict]):
        """Add training examples to the classifier"""
        for example in examples:
            doc = self.nlp.make_doc(example["text"])
            example_dict = {"cats": example["labels"]}
            self.train_examples.append(Example.from_dict(doc, example_dict))
            
    def train(self, n_iter: int = 10):
        """Train the classifier"""
        optimizer = self.nlp.begin_training()
        
        # Get names of other pipes to disable during training
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != "textcat_multilabel"]
        
        with self.nlp.disable_pipes(*other_pipes):  # Only train textcat
            for iteration in range(n_iter):
                random.shuffle(self.train_examples)
                losses = {}
                
                for example in self.train_examples:
                    self.nlp.update([example], sgd=optimizer, losses=losses)
                    
                print(f"Iteration {iteration + 1}, Losses:", losses)
    
    def predict(self, text: str) -> Dict[str, float]:
        """Predict categories for input text"""
        doc = self.nlp(text)
        predictions = {}
        for label, score in doc.cats.items():
            if score >= self.threshold:
                predictions[label] = score
        return predictions
    
    def save_model(self, path: str):
        """Save the trained model to disk"""
        self.nlp.to_disk(path)
        
    def load_model(self, path: str):
        """Load a trained model from disk"""
        self.nlp = spacy.load(path)