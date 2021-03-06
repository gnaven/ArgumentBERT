import torch
import pandas as pd
from torch.utils.data import TensorDataset, random_split
from transformers import BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


class BertData:
    # If there's a GPU available...
    if torch.cuda.is_available():    
    
        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")
    
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
    
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    
    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")    
    
    def __init__(s, TrainFile, label,score,sent_col,batch_size=16):
        df = pd.read_csv(TrainFile)
        df = df[['text','score']].dropna().reset_index()
        s.sent_col = sent_col        
        #df = df[df['label'].str.contains(label)]
        #df = df.reset_index()
        s.batch_size =batch_size
        s.df = s.sentence_trim(df)
        s.max_len =503
        s.score = s.df[score]
        
    def sentence_trim(s,df,action = 'drop'):
        
        if action =='drop':
            index_list = []
            for i in range(df.shape[0]):
                text = df[s.sent_col].iloc[i].split(' ')
                if len(text)>400:
                    index_list.append(i)
                
            df = df.drop(index = index_list,axis = 0)
        return df.reset_index()
        
        
    def tokenize_train(s,sentences,dset,score = None):
        
        if score is None:
            score = s.score
        
        # Load the BERT tokenizer.
        print('Loading BERT tokenizer...')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)        
        max_len = 0
        for sent in sentences:
        
            # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
            input_ids = tokenizer.encode(sent, add_special_tokens=True)
        
            # Update the maximum sentence length.
            max_len = max(max_len, len(input_ids))
        
        print('Max sentence length: ', max_len)          
        s.max_len = max_len
        input_ids = []
        attention_masks = []
        
        
        # For every sentence...
      
        # For every sentence...
        for sent in sentences:
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            encoded_dict = tokenizer.encode_plus(
                                sent,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = max_len,           # Pad & truncate all sentences.
                                pad_to_max_length = True,
                                return_attention_mask = True,   # Construct attn. masks.
                                return_tensors = 'pt',     # Return pytorch tensors.
                           )
            
            # Add the encoded sentence to the list.    
            input_ids.append(encoded_dict['input_ids'])
            
            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])
        
        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(score)
        
        # Print sentence 0, now as a list of IDs.
        print('Original: ', sentences[0])
        print('Token IDs:', input_ids[0]) 
        
        torch.save(input_ids,'data/'+dset+'/input_id.pt')
        torch.save(attention_masks,'data/'+dset+'/attention_mask.pt')
        torch.save(labels,'data/'+dset+'/labels.pt')        
        
        
        return input_ids, attention_masks,labels
    
    def dataloader(s,input_ids_file,attention_masks_file, labels_file):
        
        input_ids = torch.load(input_ids_file)
        attention_masks = torch.load(attention_masks_file)
        labels = torch.load(labels_file)
        
        # Combine the training inputs into a TensorDataset.
        dataset = TensorDataset(input_ids, attention_masks, labels)
              
        batch_size = s.batch_size
        
        # Create the DataLoaders for our training and validation sets.
        # We'll take training samples in random order. 
        dataloader = DataLoader(
                    dataset,  # The training samples.
                    sampler = RandomSampler(dataset), # Select batches randomly
                    batch_size = batch_size # Trains with this batch size.
                )
        
        
        return dataloader        
    
    def data_split(s,input_ids_file,attention_masks_file, labels_file):
        
        input_ids = torch.load(input_ids_file)
        attention_masks = torch.load(attention_masks_file)
        labels = torch.load(labels_file)
        
        # Combine the training inputs into a TensorDataset.
        dataset = TensorDataset(input_ids, attention_masks, labels)
        
        # Create a 90-10 train-validation split.
        
        # Calculate the number of samples to include in each set.
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        
        # Divide the dataset by randomly selecting samples.
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        print('{:>5,} training samples'.format(train_size))
        print('{:>5,} validation samples'.format(val_size))        
        
        
        # The DataLoader needs to know our batch size for training, so we specify it 
        # here. For fine-tuning BERT on a specific task, the authors recommend a batch 
        # size of 16 or 32.
        batch_size = s.batch_size
        
        # Create the DataLoaders for our training and validation sets.
        # We'll take training samples in random order. 
        train_dataloader = DataLoader(
                    train_dataset,  # The training samples.
                    sampler = RandomSampler(train_dataset), # Select batches randomly
                    batch_size = batch_size # Trains with this batch size.
                )
        
        # For validation the order doesn't matter, so we'll just read them sequentially.
        validation_dataloader = DataLoader(
                    val_dataset, # The validation samples.
                    sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
                    batch_size = batch_size # Evaluate with this batch size.
                )
        
        return train_dataloader, validation_dataloader
    
        
    
    def test_data(s,TestFile):
        # Load the dataset into a pandas dataframe.
        df = pd.read_csv(TestFile)
        df = df[['text','score']].dropna().reset_index()
        # Load the BERT tokenizer.
        print('Loading BERT tokenizer...')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True) 
        
        # Report the number of sentences.
        print('Number of test sentences: {:,}\n'.format(df.shape[0]))
        
        # Create sentence and label lists
        sentences = df['text'].values
        labels = df['score'].values
        
        # Tokenize all of the sentences and map the tokens to thier word IDs.
        input_ids = []
        attention_masks = []
        
        # For every sentence...
        for sent in sentences:
            # `encode_plus` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            #   (5) Pad or truncate the sentence to `max_length`
            #   (6) Create attention masks for [PAD] tokens.
            encoded_dict = tokenizer.encode_plus(
                                sent,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = s.max_len,           # Pad & truncate all sentences.
                                pad_to_max_length = True,
                                return_attention_mask = True,   # Construct attn. masks.
                                return_tensors = 'pt',     # Return pytorch tensors.
                           )
            
            # Add the encoded sentence to the list.    
            input_ids.append(encoded_dict['input_ids'])
            
            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])
        
        # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)
        
        torch.save(input_ids,'data/test/input_id.pt')
        torch.save(attention_masks,'data/test/attention_mask.pt')
        torch.save(labels,'data/test/labels.pt')
        
        return input_ids,attention_masks,labels
    
    def test_loader(s, input_tensor_file,atten_tensor_file,label_tensor_file):
        # Set the batch size.  
        batch_size = s.batch_size 
        
        input_ids = torch.load(input_tensor_file)
        attention_masks = torch.load(atten_tensor_file) 
        labels = torch.load(label_tensor_file)
        
        # Create the DataLoader
        prediction_data = TensorDataset(input_ids, attention_masks, labels)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)   
        
        return prediction_dataloader
        
    
    def run_all(s,TestFile, ValFile):
        train_sentences = s.df[s.sent_col]
        dfval = pd.read_csv(ValFile)
        dfval = dfval[['text','score']].dropna().reset_index()
        val_sentences = dfval[s.sent_col].values
        val_scores = dfval['score'].values
        
        s.tokenize_train(train_sentences,'train')
        s.tokenize_train(val_sentences,'val',val_scores)
        #train_dataloader, validation_dataloader = s.data_split(input_ids,attention_masks, labels)
        s.test_data(TestFile)
        
    
if __name__ == '__main__':
    
    TrainData = 'data/ann_transcript_score_class_train.csv'
    ValData = 'data/ann_transcript_score_class_val.csv'
    TestData = 'data/ann_transcript_score_class_test.csv'
    
    dataclass = BertData(TrainData, 'constructive', 'score','text')
    dataclass.run_all(TestData,ValData)
        
        