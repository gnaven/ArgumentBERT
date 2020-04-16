#################### TEST SET PREP ############################
import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import BERT_DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn

TestFile = 'data/ann_transcript_score_class_test.csv'
df = pd.read_csv(TestFile)
input_tensor_file= 'data/test/input_id.pt'
atten_tensor_file = 'data/test/attention_mask.pt'
label_tensor_file = 'data/test/labels.pt'

if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")  
    
class BERTmod(torch.nn.Module):
    
    def __init__(self,bert_out):
        
        super(BERTmod,self).__init__()
        
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=bert_out)
        self.linear = nn.Linear(bert_out, 1)
        
    def forward(self, b_input_ids,b_input_mask):
        pooled_out = self.bert(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)   
        logits = self.linear(pooled_out[0])
        
        return logits  
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True) 

dataclass = BERT_DataLoader.BertData(TestFile, 'constructive', 'score','text',batch_size=30)
prediction_dataloader = dataclass.test_loader(input_tensor_file, atten_tensor_file, label_tensor_file)


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    print(pred_flat)
    print(labels_flat)
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


#################### PREDICTION on TEST SET ######################
# Prediction on test set
"""
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 7, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)
"""

model = BERTmod(bert_out=10)

checkpoint = torch.load('savedmodels/ArgBERT_best.pt',map_location=device)
minValLoss = checkpoint['minValLoss']
model.load_state_dict(checkpoint['state_dict']) 
model.to(device)

# Put model in evaluation mode
model.eval()

# Tracking variables 
predictions , true_labels = [], []
total_test_accuracy = 0
criterion = nn.MSELoss()
# Predict 
total_test_loss=0
for batch in prediction_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
    
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    
    # Telling the model not to compute or store gradients, saving memory and 
    # speeding up prediction
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        """
        outputs = model(b_input_ids, token_type_ids=None, 
                        attention_mask=b_input_mask)
                        """
        logits = model.forward(b_input_ids, b_input_mask)
        
        loss = criterion(logits.transpose(0,1)[0],b_labels.float())
    total_test_loss += loss.item()
    #logits = outputs[0]
  
    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    
    # Store predictions and true labels
    predictions.append(logits)
    true_labels.append(label_ids)
    total_test_accuracy += flat_accuracy(logits, label_ids)

avg_test_loss = total_test_loss / len(prediction_dataloader)  
avg_test_accuracy = total_test_accuracy / len(prediction_dataloader)    

print('    DONE.')
print('test loss ', avg_test_loss)
print('test accuracy ', avg_test_accuracy)

from sklearn.metrics import matthews_corrcoef

matthews_set = []

# Evaluate each test batch using Matthew's correlation coefficient
print('Calculating Matthews Corr. Coef. for each batch...')

# For each input batch...
for i in range(len(true_labels)):
  
    # The predictions for this batch are a 2-column ndarray (one column for "0" 
    # and one column for "1"). Pick the label with the highest value and turn this
    # in to a list of 0s and 1s.
    pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
    
    # Calculate and store the coef for this batch.  
    matthews = matthews_corrcoef(true_labels[i], pred_labels_i)                
    matthews_set.append(matthews)

"""
#################### SAVING MODEL ##############################

import os

# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

output_dir = 'saved_model/'

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)


# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Good practice: save your training arguments together with the trained model
# torch.save(args, os.path.join(output_dir, 'training_args.bin'))
"""