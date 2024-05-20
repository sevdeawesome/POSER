import random
from fire import Fire
from api.model import Model
import api.util as util

def main(model_1_dir, model_2_dir, valid_data_dir):
    # model_1 = Model(model_1_dir).to("cuda")
    # model_2 = Model(model_2_dir).to("cuda")
    # valid_data = util.load_json(valid_data_dir)
    # print("Printing some outputs of the two models as a test")
    # model_1.print_generate(valid_data[0]["prompt"])
    # model_2.print_generate(valid_data[0]["prompt"])
    return random.choice([True, False]), None

if __name__ == "__main__":
    Fire(main)