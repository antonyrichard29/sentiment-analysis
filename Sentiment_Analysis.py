import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Load the dataset
sentiment_data=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/tweet_dataset.csv', encoding='unicode_escape')

# Prepare the input and output data
input_data=sentiment_data['selected_text']
output_data=sentiment_data['sentiment']

# Split the data into training and testing sets
input_data_train,input_data_test,output_data_train,output_data_test=train_test_split(input_data,output_data,test_size=0.2,random_state=42)

# Create and train the model
model=make_pipeline(TfidfVectorizer(),MultinomialNB())
model.fit(input_data_train,output_data_train)
predicted_sentiment=model.predict(input_data_test)
conf_mat=pd.DataFrame(confusion_matrix(output_data_test,predicted_sentiment),columns=['Predicted Negative','Predicted Neutral','Predicted Positive'],index=['Actual Negative','Actual Neutral','Actual Positive'])
accuracy_info=accuracy_score(output_data_test,predicted_sentiment)
print('The Accuracy is : ',accuracy_info)
print('\n-------------------------- The Confusion Matrix--------------------------')
print(conf_mat)

# Function to predict sentiment
def predict_sentiment():
  while True:
    user_input=input("\nEnter the sentence (or type 'exit' to quit):").lower()
    if user_input=='exit':
      break
    prediction=model.predict([user_input])[0]
    print(f"Sentiment is {prediction}") 

predict_sentiment()
