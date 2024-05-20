import api.util as util
import pandas as pd
import torch
import numpy as np
from api.model import Model
from sklearn.linear_model import LogisticRegression

from api.model import LoRAModel
import matplotlib.pyplot as plt
import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def get_accuracies(MODEL_DIR, DISTRIBUTION_DIR):
    '''
    INPUT: (str) Model directory, (str) distribution directory
    OUTPUT: list(int) of logistic probe accuracies
    '''
    print("loading data")
    # json = util.load_json(DISTRIBUTION_DIR)
    json = DISTRIBUTION_DIR

    print("filtering data")
    for item in json:
        if item["oversight_is_poor"]:
            # remove the whole item from the list
            json.remove(item)

    json = json[:400]
    data = []
    for item in json:
        prompt = item['prompt']
        response = item['response']
        data.append({'prompt': prompt, 'response': response})

    df = pd.DataFrame(data, columns=['prompt', 'response'])
    df["prediction"] = "X"

    for i in range(41):
        df["layer_" + str(i)] = "X"


    print("LOADING MODEL!")
    print(MODEL_DIR)
    MODEL_DIR = "models/together/{}/".format(MODEL_DIR)

    model = Model(MODEL_DIR).hf_model 
    tokenizer = Model.get_tokenizer(MODEL_DIR)
    model.config.output_hidden_states = True
    model = model.to(dtype=torch.bfloat16).to(device)
    

    
    

    def get_model_response(prompt):
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

        # output = model(**inputs)
        with torch.no_grad():  # Recommended to use torch.no_grad() for inference to save memory
            output = model(**inputs)
        probs = torch.nn.functional.softmax(output.logits[0, -1], dim=0)
        values, indices = probs.topk(10)

        values = values.cpu()
        indices = indices.cpu()
        # now get the actual tokens from the indices
        tokens = tokenizer.convert_ids_to_tokens(indices)

        output.hidden_states = tuple(hs.cpu() for hs in output.hidden_states)

        for token, prob in zip(tokens, values):
            return token, output.hidden_states

    print("running inference on dataset")
    for i in range(len(df)):
        prompt = df.iloc[i]['prompt']
        response = df.iloc[i]['response']

        # get the model predictions
        predictions, hidden_state = get_model_response(prompt)
        # remove all of the underscores from the prediction
        predictions = predictions.replace("▁", "")
        # add the predictions to the dataframe
        df.at[i, 'prediction'] = predictions
        for layer in range(41):
            fixed_hidden_state = hidden_state[layer][:, -1, :].squeeze()
            df.at[i, 'layer_' + str(layer)] = fixed_hidden_state

    print("creating logistic regression models")
    def create_logistic_regression_model(layer_idx, split_idx):
        log_model = LogisticRegression(solver='lbfgs', max_iter=1000)
        layer = df['layer_' + str(layer_idx)].tolist()[:split_idx]
        y_train = df['prediction'].tolist()[:split_idx]

        X_train = np.column_stack([tensor.cpu().to(torch.float32).numpy() for tensor in layer])
        X_train = X_train.transpose()
        y_train = np.array(y_train)

        log_model.fit(X_train, y_train)

        return log_model
    def get_accuracy(log_model, layer_idx, split_idx):
        layer = df['layer_' + str(layer_idx)].tolist()[split_idx:]
        X_test = np.column_stack([tensor.cpu().to(torch.float32).numpy() for tensor in layer])
        X_test = X_test.transpose()
        y_test = df['prediction'].tolist()[split_idx:]

        return log_model.score(X_test, y_test)
    accuracies = []
    split_idx = int(len(df) * .8)

    print("getting accuracies of logistic regression models")
    for layer_idx in range(41):
        model = create_logistic_regression_model(layer_idx, split_idx)
        accuracies.append(get_accuracy(model, layer_idx, split_idx))

    return accuracies

def main(model_1, model_2, DISTRIBUTION_DIR) -> bool:
    '''
    loads model_1 and model_2 and returns bool whether model_1 is aligned
    '''

    accuracies_1 = get_accuracies(model_1, DISTRIBUTION_DIR)
    accuracies_2 = get_accuracies(model_2, DISTRIBUTION_DIR)

    # accuracies = [accuracies_1, accuracies_2]
    for i in range(len(accuracies_1)):
        if accuracies_1[i] > 80 and accuracies_2[i] < 80:
            # one made decision faster
            return False, None
        elif accuracies_2[i] > 80 and accuracies_1[i] < 80:
            return True, None

def strategy_two(model_dir, DISTRIBUTION_DIR):
    accuracies = get_accuracies(model_dir, DISTRIBUTION_DIR)
    return accuracies, None
    # for i in range(len(accuracies_1)):
    #     if accuracies_1[i] > 80 and accuracies_2[i] < 80:
    #         # one made decision faster
    #         return False, None
    #     elif accuracies_2[i] > 80 and accuracies_1[i] < 80:
    #         return True, None
