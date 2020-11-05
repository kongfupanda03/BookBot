'''
Initializing steps for the REST API:
1. create virtual environment
2. install dependencies:
    pip install -r requirements.txt
3. download and install ngrok (need to resolve permission issue in macos)
    https://ngrok.com/download
4. copy the ngrok file to project directory
5. lauch the python flask script "server.py" to run
6. launch ngrok:
    1) open terminal and cd to project directory
    2) run: ./ngrok http 5000
    3) copy the public url and paste in dialogflow fufilment Webhook URL
7. In case of any error regarding address already in use:
    run: ps -fA | grep python
    then: kill xxxxx
'''

from flask import Flask, request, make_response, jsonify
import pandas as pd
from transformers import BertModel, BertTokenizer
import torch
from torch import nn
import time
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PRE_TRAINED_MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 512
class_names =['negative','positive']

class SentimentClassifier(nn.Module):  
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.5)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return self.out(output)

def process_request(req):
    query_result = req.get('queryResult')
    query_text = query_result.get('queryText', None)
    output_context = query_result.get('outputContexts', None)
    params = query_result.get('parameters', None)
    intent_display_name = query_result.get('intent', None).get('displayName')
    return query_text, params, intent_display_name, output_context

def get_orginal_list(df):
    orgianl_list = []
    for i in range(3):
        bookID = df['BookID'][i]
        orgianl_list.append(bookID)
    return orgianl_list

def get_book_details(book_list,col_name):
    book_details_list = []
    for item in book_list:
        bookID = str(item)
        book_detail = recommend_df[recommend_df['BookID'] == bookID][[col_name]]\
        .drop_duplicates()\
        .reset_index(drop=True)[col_name][0]
        book_details_list.append(book_detail)
    return book_details_list

def get_recommended_list(df, book_index,rank):
    recommended_list = []
    for i in range(5):
        col_name = rank + '_'+ str(i+1) +'_id'
        bookID = str(df[col_name][book_index])
        recommended_bookID = recommend_df[recommend_df['BookID'] == bookID][['BookID']]\
        .drop_duplicates()\
        .reset_index(drop=True)['BookID'][0]
        recommended_list.append(recommended_bookID)
    return recommended_list

def first_round_recommendation(topic,genre):
    recommend_df_top10 = recommend_df[(recommend_df['Topic'] == topic) & (recommend_df['Genre'] == genre)].nlargest(10, 'AverageRating').reset_index(drop=True)
    recommend_df_random3 = recommend_df_top10.sample(3).reset_index(drop=True)
    recommend_df_random3.to_csv('recommend_df_random3.csv',index=False)
    orgianl_list = get_orginal_list(recommend_df_random3)
    book_title_list = get_book_details(orgianl_list,'BookTitle')
    book_URL_list = get_book_details(orgianl_list,'BookURL')
    book_emo1_list = get_book_details(orgianl_list,'top1_emo')
    book_emo2_list = get_book_details(orgianl_list,'top2_emo')
    book_pic_list = get_book_details(orgianl_list,'PictureURL')
    book_keywords_list = get_book_details(orgianl_list,'keywords')     
    my_result = {
        "fulfillmentMessages": [
            {
                "text": {
                    "text": [
                        "Here are some books you might be insterested in."
                    ]
                }
            },
            {
                "card": {
                    "title": book_title_list[0],
                    "subtitle": 'Main emotions: ' + str(book_emo1_list[0]) + ', ' + str(book_emo2_list[0]),
                    "imageUri": book_pic_list[0],
                    "buttons": [
                        {
                            "text": book_keywords_list[0],
                            "postback": book_URL_list[0]
                        }
                     ]
                }
            },
            {
                "card": {
                    "title": book_title_list[1],
                    "subtitle": 'Main emotions: ' + str(book_emo1_list[1]) + ', ' + str(book_emo2_list[1]),
                    "imageUri": book_pic_list[1],
                    "buttons": [
                        {
                            "text": book_keywords_list[1],
                            "postback": book_URL_list[1]
                        }
                     ]
                }
            },
            {
                "card": {
                    "title": book_title_list[2],
                    "subtitle": 'Main emotions: ' + str(book_emo1_list[2]) + ', ' + str(book_emo2_list[2]),
                    "imageUri": book_pic_list[2],
                    "buttons": [
                        {
                            "text": book_keywords_list[2],
                            "postback": book_URL_list[2]
                        }
                     ]
                }
            },
            {
                "card": {
                    "title": "May I know whether you have read any of these books? If yes, would you like to provide a review for a customized recommendation?",
                    "buttons": [
                        {
                            "text": "Y"
                            },
                        {
                            "text": "N"
                            }
                        ]
                }
            }
        ]
    }
    return my_result

def second_round_recommendation(df, book_index, rank):
    recommended_list = get_recommended_list(df,book_index,rank)
    orgianl_list = get_orginal_list(df)
    second_round_list = [x for x in recommended_list if x not in orgianl_list]
    second_round_list_3 = random.sample(second_round_list, 3)
    book_title_list = get_book_details(second_round_list_3,'BookTitle')
    book_URL_list = get_book_details(second_round_list_3,'BookURL')
    book_emo1_list = get_book_details(second_round_list_3,'top1_emo')
    book_emo2_list = get_book_details(second_round_list_3,'top2_emo')
    book_pic_list = get_book_details(second_round_list_3,'PictureURL')
    book_keywords_list = get_book_details(second_round_list_3,'keywords')
    my_result = {
    "fulfillmentMessages": [
        {
            "text": {
                "text": [
                    "Please find below books you might be interested in based on your review."
                ]
            }
        },
        {
            "card": {
                "title": book_title_list[0],
                "subtitle": 'Main emotions: ' + str(book_emo1_list[0]) + ', ' + str(book_emo2_list[0]),
                "imageUri": book_pic_list[0],
                "buttons": [
                    {
                        "text": book_keywords_list[0],
                        "postback": book_URL_list[0]
                    }
                 ]
            }
        },
        {
            "card": {
                "title": book_title_list[1],
                "subtitle": 'Main emotions: ' + str(book_emo1_list[1]) + ', ' + str(book_emo2_list[1]),
                "imageUri": book_pic_list[1],
                "buttons": [
                    {
                        "text": book_keywords_list[1],
                        "postback": book_URL_list[1]
                    }
                 ]
            }
        },
        {
            "card": {
                "title": book_title_list[2],
                "subtitle": 'Main emotions: ' + str(book_emo1_list[2]) + ', ' + str(book_emo2_list[2]),
                "imageUri": book_pic_list[2],
                "buttons": [
                    {
                        "text": book_keywords_list[2],
                        "postback": book_URL_list[2]
                    }
                 ]
            }
        },
        {
            "text": {
                "text": [
                    "Thanks for your support! Feel free to explore more on other books!"
                ]
            }
        }
    ]
    }
    return my_result

app = Flask(__name__)

@app.route('/', methods=['POST'])
def handler():
    req = request.get_json(silent=True, force=True)
    req_info = process_request(req)
    if req_info[2] == 'Detect Genre Intent':
        print('req_info = ', req_info)
        genre = req_info[1]['genres']
        print('Genre: ', genre)
        topic_list = genre_topic_combo_df[genre_topic_combo_df['Genre'] == genre]['Topic'].to_list()
        text_response_1 = 'Could you please tell me which one of the following topics in the ' + genre + ' genre you would like to further dig into?'
        my_result = {
            "fulfillmentMessages": [
                {
                    "card": {
                        "title": text_response_1,
                        "buttons": [
                            {
                                "text": topic_list[0] 
                                },
                            {
                                "text": topic_list[1] 
                                },
                            {
                                "text": topic_list[2] 
                                }]
                        }
                    }
            ]
        }
        res = jsonify(my_result)
        print('my_result = ', my_result)
        r = make_response(res)
    elif req_info[2] == 'Detect Topic Intent':
        print('req_info = ', req_info)
        topic = req_info[1]['topics']
        print('Topic: ', topic)
        genre = req_info[3][0]['parameters']['genres']
        print(genre)
        my_result = first_round_recommendation(topic,genre)
        res = jsonify(my_result)
        print('my_result = ', my_result)
        r = make_response(res)
    elif req_info[2] == 'Detect Review Intent':
        print('req_info = ', req_info)
        recommend_df_random3 = pd.read_csv('recommend_df_random3.csv')
        if req_info[1]['review'] == 'N':
            my_result = {
                "fulfillmentMessages": [
                    {
                        "text": {
                            "text": [
                                "Thank you for using BookBot! Have a great day ahead!"
                            ]
                        }
                    }
                ]
            }
            res = jsonify(my_result)
            r = make_response(res)
        elif req_info[1]['review'] == 'Y': 
            my_result = {
                "fulfillmentMessages": [
                    {
                    "card": {
                        "title": "Please choose one book you would like to leave a review.",
                        "buttons": [
                            {
                                "text": 'Book1' 
                                },
                            {
                                "text": 'Book2'
                                },
                            {
                                "text": 'Book3'
                                }]
                        }
                    },
                    {
                        "text": {
                            "text": [
                                'Book1: ' + recommend_df_random3['BookTitle'][0]
                            ]
                        }
                    },
                    {
                        "text": {
                            "text": [
                                'Book2: ' + recommend_df_random3['BookTitle'][1]
                            ]
                        }
                    },
                    {
                        "text": {
                            "text": [
                                'Book3: ' + recommend_df_random3['BookTitle'][2]
                            ]
                        }
                    }
                ]
                }
            res = jsonify(my_result)
            print('my_result = ', my_result)
            r = make_response(res)     
    elif req_info[2] == 'Choose Book to Review Intent':
        print('req_info = ', req_info)
        book_number = req_info[1]['book_index']
        book_index = int(book_number) - 1
        recommend_df_random3 = pd.read_csv('recommend_df_random3.csv')
        book_title = recommend_df_random3['BookTitle'][book_index]
        my_result = {
            "fulfillmentMessages": [
                {
                    "text": {
                        "text": [
                            "Please input your review for Book" + book_number  + ": " + book_title + "."
                        ]
                    }
                }                
            ]
        }
        res = jsonify(my_result)
        print('my_result = ', my_result)
        r = make_response(res) 
    elif req_info[2] == 'User Input - fallback':
        print('req_info = ', req_info)
        user_input = req_info[0]
        book_index = int(req_info[3][0]['parameters']['book_index']) - 1
        
        start = time.process_time()
        encoding = tokenizer.encode_plus(
        user_input,
        add_special_tokens=True,
        max_length=MAX_LEN,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True)
    
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        output = model(input_ids, attention_mask)
        _, prediction = torch.max(output, dim=1)
        semtiment_polarity = class_names[prediction]

        print("semtiment_polarity: ",semtiment_polarity)
        print('response time: ',time.process_time() - start)
        
        recommend_df_random3 = pd.read_csv('recommend_df_random3.csv')
        
        if semtiment_polarity == 'positive':
            my_result = second_round_recommendation(recommend_df_random3, book_index, 'Top')
            print('my_result = ', my_result)
            res = jsonify(my_result)
            r = make_response(res)
        elif semtiment_polarity == 'negative':
            my_result = second_round_recommendation(recommend_df_random3, book_index, 'Last')
            print('my_result = ', my_result)
            res = jsonify(my_result)
            r = make_response(res)
    return r


if __name__ == "__main__":
    genre_topic_combo_df = pd.read_csv('PLP_BookBot_Genre_Topic_Combo.csv')
    recommend_df = pd.read_csv('PLP_BookBot_Processed_Dataset.csv') 
    model = SentimentClassifier(2)
    model.load_state_dict(torch.load('best_model_state_512_lr5e-6.bin',map_location=torch.device('cpu')))
    model = model.to(device)
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME,do_lower_case=True)
    app.run(debug=True)
