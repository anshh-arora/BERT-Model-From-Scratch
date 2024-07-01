# BERT(Bidirectional Encoder Representions from Transformers)
# Building from Scratch

import torch
import torch.nn as nn
import torch.optim as optim

# Constants
vocab = {'<pad>': 0, 'the': 1, 'cat': 2, 'sat': 3, 'on': 4, 'mat': 5, 'dog': 6, 'slept': 7, 'rug': 8, 'bird': 9, 'flew': 10, 'away': 11}  # Extended vocabulary
vocab_size = len(vocab) # Calculate the length of the vocabulary
print(f"vocab_size :{vocab_size}")

embedding_dim = 10  # Dimension of the word embeddings (in simple words this is total number of vocab)
hidden_dim = 12  # Dimension of the LSTM hidden layer (size of the hidden layer)
learning_rate = 0.01  # Learning rate for the optimizer
num_epochs = 100  # Number of training epochs

# Sample training data (we will train the bert model on this data)
training_data = [
    "the cat sat on the mat",
    "the dog slept on the rug",
    "the bird flew away"
]

# Tokenizer function
def tokenize_text(text, vocab):
    tokens = text.lower().split()  # Convert text to lowercase and split into words
    token_ids = [vocab[token] for token in tokens if token in vocab]  # Map the words to their corresponding IDs
    return token_ids #return the token ids for eg: [The]:1, [cat]:2

"""
for creating dataset we are using torch.tensor in  input sequence we exclude the last token and in target we exclude the first token 
input -> "The" : 1, "Cat" : 2, "Sat" : 3, "On" : 4, "The" : 1 , "Mat" (we exclude the mat in input)
target-> "Cat" : 2, "Sat" : 3, "On" : 4, "The" : 1 , "Mat" : 5, (we exclude the first token here)
"""
# Now We Create a dataset 
tokenized_texts = [tokenize_text(text, vocab) for text in training_data]  # Tokenize each sentence in the training data
input_sequences = [torch.tensor(text[:-1], dtype=torch.long) for text in tokenized_texts if len(text) > 1]  # Create input sequences (exclude last token)
target_sequences = [torch.tensor(text[1:], dtype=torch.long) for text in tokenized_texts if len(text) > 1]  # Create target sequences (exclude first token)

"""
Before Defining Model we sholud know what does "nn.embedding" , "nn.LSTM", "nn.Linear", "Softmax Function" do
1) nn.embedding -> convert word indicies into dense vector (multiply one hot encoding with randomly initialiize embedding matrix and findout the hidden layer output)

2) nn.LSTM -> Process the sequence of embedding and Capture dependences
in simple words we can say that "LSTM" learn from previous time stamps and update the the current state.

3) nn.linear -> It is used to apply a linear transformation to the incoming data, converting it to a different shape. 
|_ It is used to convert the LSTM hidden layer state to a prediction of the next word in the vocabulary.

4) Softmax function -> Convert the output into probability. in the range of (0 to 1).

"""
# Model definition
class NextWordPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(NextWordPredictor, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # Embedding layer to convert word IDs to embeddings
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # LSTM layer for sequence modeling
        self.linear = nn.Linear(hidden_dim, vocab_size)  # Linear layer to map LSTM outputs to vocabulary size

    def forward(self, x):
        embeds = self.embedding(x)  # Convert input token IDs to embeddings
        lstm_out, _ = self.lstm(embeds)  # Process embeddings through the LSTM layer
        output = self.linear(lstm_out)  # Map LSTM outputs to vocabulary size
        return output

    def predict_next_token(self, input_token_ids, temperature=1.0):
        # Predict next token probabilities
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation for prediction
            embeds = self.embedding(input_token_ids)  # Convert input token IDs to embeddings
            lstm_out, _ = self.lstm(embeds)  # Process embeddings through the LSTM layer
            output = self.linear(lstm_out[:, -1, :])  # Focus on the last output
            softmax_output = nn.functional.softmax(output / temperature, dim=-1)  # Apply softmax to get probabilities and find out the correct predicted word
        return softmax_output
"""
1) nn.crossEntropyLoss() -> Computes the loss between the predicted probability and the actual values
  formlua  => Loss = -E y_actuallog(y_pred)

2) Adam -> It is a optimization algorithm that can be used instead of "Classical Stochastic Gradient Descent(SGD)" to update weights iteratively based on training data.
 ! It involve a combination of two gradient descent methodologies:-
 (i) Momentum :- It is used to accelerates the gradient descent algo by taking into consederation the "Exponential Weight Average" of the Gradient. using average make the algo converge toward the minimum in a faster pace.
 (ii) RMSP :- Root Mean Square Propogation is an adaptive learing algo that tries to improve AdaGrad.
"""
# Instantiate the model and define loss function and optimizer
model = NextWordPredictor(vocab_size, embedding_dim, hidden_dim)  # Instantiate the model
criterion = nn.CrossEntropyLoss()  # Define the loss function (cross-entropy loss)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Define the optimizer (Adam)

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    total_loss = 0  # Initialize total loss for the epoch

    for input_seq, target_seq in zip(input_sequences, target_sequences):
        optimizer.zero_grad()  # Clear previous gradients
        output = model(input_seq.unsqueeze(0))  # Get the model's output for the input sequence (add batch dimension to a tensor specificaly at position 0)
        loss = criterion(output.squeeze(0), target_seq)  # Compute the loss between model output and target sequence
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update the model parameters

        total_loss += loss.item()  # Accumulate the loss

    if (epoch + 1) % 10 == 0:  # Print loss every 10 epochs
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(input_sequences):.4f}")

print(f"TEXT : {training_data}")
# Evaluation and prediction example
model.eval()  # Set the model to evaluation mode
# it ensures that the model behaves consistently during both training and inference.


# Example input text
input_text = input("enter a sentence from the above text to check the bert model: " )

# Tokenize input text
token_ids = tokenize_text(input_text, vocab)  # Tokenize the input text
if len(token_ids) > 0:
    input_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)  # Convert token IDs to a tensor and add batch dimension

    # Predict next token probabilities
    softmax_output = model.predict_next_token(input_tensor)  # Get the model's prediction

    # Sample from the softmax output using temperature (optional)
    temperature = 0.8  # Adjust temperature to control the randomness of sampling
    predicted_token_indices = torch.multinomial(softmax_output, num_samples=1)  # Sample from the probability distribution
    predicted_token_id = predicted_token_indices[0].item()  # Extract the index of the first sample
    predicted_token = list(vocab.keys())[list(vocab.values()).index(predicted_token_id)]  # Convert the token ID back to a word
    # Display results
    print(f"Input Text: {input_text}")  # user need to input text 
    print(f"Tokenized Text: {token_ids}")  # Display the tokenized input text
    print(f"Predicted Next Token: {predicted_token}")  # Display the predicted next token
else:
    print("Input sequence is empty or too short. Please provide a valid input.")  # Handle invalid input sequences
