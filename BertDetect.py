import logging
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel, EncoderDecoderModel, ReformerModelWithLMHead, ReformerTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt 
import os,sys,random,time,shutil,warnings

from sklearn.metrics import accuracy_score, f1_score
from utils import _generate_past_question_correctness_info

def remove_folder(folder_path):
    try:
        shutil.rmtree(folder_path)  # Use shutil.rmtree to remove the directory and its contents
        print(f"Folder '{folder_path}' and all its contents have been removed successfully.")
    except FileNotFoundError:
        print(f"The folder '{folder_path}' does not exist.")
    except Exception as e:
        print(f"An error occurred while removing the folder: {e}")



class CustomDataset(Dataset):
    def __init__(self, dataframe, dataframe_raw, tokenizer, max_length):
        self.dataframe = dataframe
        self.dataframe_raw = dataframe_raw
        self.tokenizer = tokenizer
        self.max_length = max_length

        if tokenizer.pad_token_id is None: tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        input_prompt_1 = self.dataframe.iloc[idx]['input_past_prompt']+self.dataframe.iloc[idx]['input_future_prompt']
        question_id = self.dataframe.iloc[idx]['question_id']
        llm_predict_answer = self.dataframe.iloc[idx]['future_correctness_predict']
        assert llm_predict_answer in [0,1]
        if llm_predict_answer == 1:
            llm_predict_answer_str = f'The student will answer this question {question_id} correctly. Reasons: ' 
        else:
            llm_predict_answer_str = f'The student will answer this question {question_id} wrongly. Reasons: ' 
        input_prompt_2 = llm_predict_answer_str + self.dataframe.iloc[idx]['llm_response']

        input_token_len = len(self.tokenizer.tokenize(input_prompt_1))+len(self.tokenizer.tokenize(input_prompt_2))
        if input_token_len >= 512:
            # print('input_token_len too long: ',input_token_len,' We have to remove choice contents.')
            sample_uid = self.dataframe.iloc[idx]['uid']
            sample_data = self.dataframe_raw[self.dataframe_raw['uid']==sample_uid]
            input_past_prompt_no_choice = _generate_past_question_correctness_info(sample_data,include_choice=False)
            input_prompt_1 = input_past_prompt_no_choice+self.dataframe.iloc[idx]['input_future_prompt']



        label = self.dataframe.iloc[idx]['llm_correctness']
        
        inputs = self.tokenizer(input_prompt_1, input_prompt_2, truncation='longest_first', padding='max_length', max_length=self.max_length, return_tensors='pt')

        item = {key: val.squeeze() for key, val in inputs.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        
        

        return item


class Experiment_Pipeline():
    def __init__(self, max_length, log_folder, dataset_raw_path, load_model_type, random_seed = 4, lr=1e-5):
        self.max_length = max_length
        self.set_seed(random_seed)

        self.dataframe_raw = pd.read_csv(dataset_raw_path,sep='\t')

        self.log_folder = log_folder
        self.load_model_type = load_model_type

        self.model_init(load_model_type,lr)

    def set_seed(self,seed_num):
        np.random.seed(seed_num)
        random.seed(seed_num)
        torch.manual_seed(seed_num)
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)  
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False

    def dataset_prepare(self,dataset_path,batch_size = 16):
        dataframe = pd.read_csv(dataset_path,sep='\t')
        dataframe_train = dataframe[dataframe['data_type']=='train']
        dataframe_test = dataframe[dataframe['data_type']=='test']
        
        dataset = CustomDataset(dataframe_train, self.dataframe_raw, self.tokenizer, self.max_length)
        test_dataset = CustomDataset(dataframe_test, self.dataframe_raw, self.tokenizer, self.max_length)
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size 
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        self.train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
        self.val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        self.test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


    def model_save(self,checkpoint,checkpoint_path,tokenizer,tokenizer_path,e2emodel,e2emodel_path):
        if os.path.exists(checkpoint_path): os.remove(checkpoint_path)
        torch.save(checkpoint, checkpoint_path)
        if os.path.exists(tokenizer_path): remove_folder(tokenizer_path)
        tokenizer.save_pretrained(tokenizer_path)
        if os.path.exists(e2emodel_path): remove_folder(e2emodel_path)
        e2emodel.save_pretrained(e2emodel_path)


    def model_init(self,load_model_type,lr=1e-4):
        assert load_model_type in ['best','last','none']

        self.checkpoint_last_path = self.log_folder + '/model_last.pt'
        self.checkpoint_best_path = self.log_folder + '/model_best.pt'
        self.tokenizer_last_path = self.log_folder + '/tokenizer_last'
        self.tokenizer_best_path = self.log_folder + '/tokenizer_best'
        self.e2emodel_last_path = self.log_folder + '/e2emodel_last'
        self.e2emodel_best_path = self.log_folder + '/e2emodel_best'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if load_model_type in ['best','last']:
            self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_last_path) if load_model_type == 'last' else BertTokenizer.from_pretrained(self.tokenizer_best_path)
            self.model = BertForSequenceClassification.from_pretrained(self.e2emodel_last_path) if load_model_type == 'last' else BertForSequenceClassification.from_pretrained(self.e2emodel_best_path)
            checkpoint = torch.load(self.checkpoint_last_path) if load_model_type == 'last' else torch.load(self.checkpoint_best_path)
            self.epoch_exist = checkpoint['epoch']
            # self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_losses = checkpoint['train_loss']
            self.val_losses = checkpoint['val_loss']
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(self.device)
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
            self.epoch_exist = 0
            self.train_losses = []
            self.val_losses = []

        # self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
        # self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

    def model_train(self, epochs=3, vis=True):
        loss_file = self.log_folder+'/loss.csv'
        with open(loss_file, "a+") as file1:
            file1.write('train_loss,val_loss,accuracy,f1\n')

        for epoch in range(epochs):
            if epoch + 1 <= self.epoch_exist: continue
            self.model.train()
            total_loss = 0
            time_train_start = time.time()
            for batch in self.train_dataloader:
                self.optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # outputs = self.model(
                #     input_ids=input_ids,
                #     attention_mask=attention_mask,
                #     labels=labels
                # )

                # print('outputs: ',outputs)

                # loss = outputs.loss

                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs.logits, labels)
                # loss = self.criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            time_train_end = time.time()

            avg_loss = total_loss / len(self.train_dataloader)
            self.train_losses.append(avg_loss)
            train_time = time_train_end - time_train_start
            print(f'Epoch {epoch + 1}, Training loss: {avg_loss}, Training Time: {train_time}')
            
            val_loss, accuracy, f1 = self.model_eval(eval_mode='val')

            with open(loss_file, "a+") as file1:
                file1.write(str(avg_loss)+','+str(val_loss)+','+str(accuracy)+','+str(f1)+'\n')

            checkpoint = {
                'epoch': epoch + 1,
                # 'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': self.train_losses,
                'val_loss': self.val_losses
            }
            
            self.model_save(checkpoint,self.checkpoint_last_path,self.tokenizer,self.tokenizer_last_path,self.model,self.e2emodel_last_path)

            if (len(self.val_losses) == 0) or (len(self.val_losses) != 0 and val_loss == min(self.val_losses)):
                self.model_save(checkpoint,self.checkpoint_best_path,self.tokenizer,self.tokenizer_best_path,self.model,self.e2emodel_best_path)

        if vis == True:
            fig, ax = plt.subplots()
            plt.plot(self.train_losses, label='Train Loss')
            plt.plot(self.val_losses, label='Validation Loss')
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training and Validation Loss")
            plt.legend()
            plt.savefig(self.log_folder+'/loss.png')

    def model_eval(self,eval_mode):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        assert eval_mode in ['test','val']
        dataloader_part = self.test_dataloader if eval_mode == 'test' else self.val_dataloader
        time_eval_start = time.time()

        with torch.no_grad():
            for batch in dataloader_part:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                # outputs = self.model(
                #     input_ids=input_ids,
                #     attention_mask=attention_mask,
                #     labels=labels
                # )
                # loss = outputs.loss
                loss = self.criterion(outputs.logits, labels)
                # loss = self.criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                total_loss += loss.item()

                # Calculate accuracy
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        time_eval_end = time.time()
        time_eval = time_eval_end - time_eval_start
        avg_loss = total_loss / len(dataloader_part)
        self.val_losses.append(avg_loss)

        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')

        print(f'{eval_mode} loss: {avg_loss}, accuracy: {accuracy}, F1 score: {f1}, time: {time_eval}')
        return avg_loss, accuracy, f1


    def predict_from_dataframe(self, dataframe):
        dataset = CustomDataset(dataframe, self.dataframe_raw, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=1)  # Use batch size 1 for single sample prediction

        predictions = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)

                outputs = self.model(input_ids, attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                predictions.extend(preds)

        return predictions




def run_exp_bertDetect():
    dataset_path = '/home/songlin/dataset/GKT/eduagent/dataset_gkt_bert_detect_llm_KT.csv'
    dataset_raw_path = '/home/songlin/dataset/GKT/eduagent/dataset_gkt_split.csv'
    log_folder = '/home/songlin/study_results/GKT/bertDetect/'
    
    experiment_pipeline = Experiment_Pipeline(512,log_folder,dataset_raw_path,load_model_type='none',lr=5e-6)
    experiment_pipeline.dataset_prepare(dataset_path)
    experiment_pipeline.model_train(epochs=100,vis=True)
    experiment_pipeline.model_eval(eval_mode='test')

def bert_detect_llm_KT():
    dataset_path = '/home/songlin/dataset/GKT/eduagent/dataset_gkt_bert_detect_llm_KT.csv' 
    dataset_raw_path = '/home/songlin/dataset/GKT/eduagent/dataset_gkt_split.csv'
    output_path = '/home/songlin/study_results/GKT/bertDetect/final_predict.csv'
    bertDetect_folder = '/home/songlin/study_results/GKT/bertDetect/'

    dataset = pd.read_csv(dataset_path,sep='\t')
    dataset_test = dataset[dataset['data_type']=='test']
    student_list = list(set(dataset_test['student_id']))
    student_list.sort()
    label_list = []
    predict_list = []
    llm_predict_list = []

    pipeline = Experiment_Pipeline(max_length=512, log_folder=bertDetect_folder, dataset_raw_path=dataset_raw_path, load_model_type='best')

    for student_id in student_list:
        print(f'student: {student_id}')
        dataset_test_student = dataset_test[dataset_test['student_id']==student_id]
        question_list = list(set(dataset_test_student['question_id']))
        for question_id in question_list:
            data_item = dataset_test_student[dataset_test_student['question_id']==question_id]
            label = data_item['future_label'].values[0]
            llm_predict = data_item['future_correctness_predict'].values[0]
            bert_detect = pipeline.predict_from_dataframe(data_item)
            assert bert_detect[0] in [0,1]
            final_predict = llm_predict if bert_detect == 1 else 1-llm_predict
            label_list.append(label)
            predict_list.append(final_predict)
            llm_predict_list.append(llm_predict)
    accuracy = accuracy_score(label_list, predict_list)
    f1 = f1_score(label_list, predict_list, average='weighted')
    llm_accuracy = accuracy_score(label_list, llm_predict_list)
    llm_f1 = f1_score(label_list, llm_predict_list, average='weighted')
    print(f'LLM+Bert accuracy: {accuracy}, f1: {f1}')        
    print(f'LLM only accuracy: {llm_accuracy}, f1: {llm_f1}')        

run_exp_bertDetect()
bert_detect_llm_KT()
