import random
import json
import torch
from model import *
from nltk_utils import *
from torch.utils.data import Dataset, DataLoader

class ChatDataset(Dataset):
    """Dataset class for chatbot training."""
    def __init__(self, X_train, y_train):
        self.x_data = X_train
        self.y_data = y_train
        self.n_samples = len(X_train)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

def prepare_data(intents_file):
    """Prepares the training data from the intents JSON file."""
    with open(intents_file, 'r') as f:
        intents = json.load(f)

    all_words = []
    tags = []
    xy = []

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            words = tokenize(pattern)
            all_words.extend(words)
            xy.append((words, tag))

    ignore_words = ['?', '.', '!']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    # Create training data
    X_train = []
    y_train = []
    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        label = tags.index(tag)
        y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return all_words, tags, X_train, y_train

def train_model(X_train, y_train, input_size, hidden_size, output_size, num_epochs=1000, batch_size=16, learning_rate=0.001):
    """Trains the neural network model."""
    dataset = ChatDataset(X_train, y_train)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for words, labels in train_loader:
            words, labels = words.to(device), labels.to(dtype=torch.long).to(device)
            
            # Forward pass
            outputs = model(words)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model, criterion, optimizer

def save_model(model, input_size, hidden_size, output_size, all_words, tags, filename='data.pth'):
    """Saves the trained model to a file."""
    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags
    }
    torch.save(data, filename)
    print(f'Model saved to {filename}')

def load_model(filename='data.pth'):
    """Loads a trained model from a file."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load(filename)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]

    model = NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    return model, input_size, hidden_size, output_size, all_words, tags

def chatbot_response(model, sentence, all_words, tags, intents, device):
    """Generates a response from the chatbot given a sentence."""
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, -1)
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.5:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    else:
        return "I'm sorry, I couldn't understand that. Please provide more details."


def main():
    """Main function to train and run the chatbot."""
    intents_file = 'intents.json'
    all_words, tags, X_train, y_train = prepare_data(intents_file)

    import json

    with open('intents.json') as file:
        intents = json.load(file)


    input_size = len(X_train[0])
    hidden_size = 8
    output_size = len(tags)

    # Train the model
    model, criterion, optimizer = train_model(X_train, y_train, input_size, hidden_size, output_size)

    # Save the model
    save_model(model, input_size, hidden_size, output_size, all_words, tags)

    # Load the trained model
    model, input_size, hidden_size, output_size, all_words, tags = load_model()

    # Run the chatbot
    bot_name = "PlantBot"
    print("Welcome to PlantBot! Ask about plant diseases and remedies. (Type 'quit' to exit)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    while True:
        sentence = input("You: ")
        if sentence.lower() == "quit":
            print("Goodbye! Stay safe and take care of your plants.")
            break
    
        # Process input and get response
        response = chatbot_response(model, sentence, all_words, tags, intents, device)
        print(f"PlantBot: {response}")

if __name__ == "__main__":
    main()
