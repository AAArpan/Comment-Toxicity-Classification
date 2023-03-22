# Comment-Toxicity-Classification
Recurrent Neural Network based classification model which can be used to detect toxic and offensive comments.
The model is typically trained on a Kaggle dataset (https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) of comments that have been manually labeled in 6 categories **(toxic, severe_toxic, obscene, threat, insult and identity_hate)**, and it uses this data to learn patterns and relationships that can help it accurately classify new comments.

Model used for this task is **Bidirectional LSTM (Long-Short-Term-Memory)** model.

# ADVANTAGES 
1. A Bidirectional Long Short-Term Memory (LSTM) is a type of recurrent neural network that processes sequential data in both forward and backward directions. Unlike a traditional LSTM, which processes data in only one direction, a bidirectional LSTM can take advantage of information from both past and future data points to make more accurate predictions.

      <img width="250" alt="6" src="https://user-images.githubusercontent.com/108794407/226996149-0c4c6039-ed6c-409f-a549-e7613ca9fcf4.png">

2. In a bidirectional LSTM, the input data is split into two parts, one for the forward direction and one for the backward direction. Each part is processed separately by a separate LSTM layer, and the outputs of the two layers are combined at each time step to form the final output of the model. This allows the model to capture dependencies between both past and future data points, which can be useful in a variety of applications, such as speech recognition, machine translation, and sentiment analysis.

3. Bidirectional LSTMs are particularly useful when working with long sequences of data, as they can capture long-term dependencies and avoid the problem of vanishing gradients that can occur with traditional LSTMs. However, they are also more computationally expensive than traditional LSTMs, as they require processing the data twice in opposite directions.
