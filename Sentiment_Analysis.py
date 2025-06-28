{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMPC2he/ub5FVWCYUJpLeNJ",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/antonyrichard29/sentiment-analysis/blob/main/Sentiment_Analysis.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "d97hpBXo5yzU"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.naive_bayes import MultinomialNB\n",
        "from sklearn.pipeline import make_pipeline\n",
        "from sklearn.metrics import confusion_matrix\n",
        "from sklearn.metrics import accuracy_score\n",
        "\n",
        "# Load the dataset\n",
        "sentiment_data=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/tweet_dataset.csv', encoding='unicode_escape')\n",
        "\n",
        "# Prepare the input and output data\n",
        "input_data=sentiment_data['selected_text']\n",
        "output_data=sentiment_data['sentiment']\n",
        "\n",
        "# Split the data into training and testing sets\n",
        "input_data_train,input_data_test,output_data_train,output_data_test=train_test_split(input_data,output_data,test_size=0.2,random_state=42)\n",
        "\n",
        "# Create and train the model\n",
        "model=make_pipeline(TfidfVectorizer(),MultinomialNB())\n",
        "model.fit(input_data_train,output_data_train)\n",
        "predicted_sentiment=model.predict(input_data_test)\n",
        "conf_mat=pd.DataFrame(confusion_matrix(output_data_test,predicted_sentiment),columns=['Predicted Negative','Predicted Neutral','Predicted Positive'],index=['Actual Negative','Actual Neutral','Actual Positive'])\n",
        "accuracy_info=accuracy_score(output_data_test,predicted_sentiment)\n",
        "print('The Accuracy is : ',accuracy_info)\n",
        "print('\\n-------------------------- The Confusion Matrix--------------------------')\n",
        "print(conf_mat)\n",
        "\n",
        "# Function to predict sentiment\n",
        "def predict_sentiment():\n",
        "  while True:\n",
        "    user_input=input(\"\\nEnter the sentence (or type 'exit' to quit):\").lower()\n",
        "    if user_input=='exit':\n",
        "      break\n",
        "    prediction=model.predict([user_input])[0]\n",
        "    print(f\"Sentiment is {prediction}\")\n",
        "\n",
        "predict_sentiment()"
      ]
    }
  ]
}