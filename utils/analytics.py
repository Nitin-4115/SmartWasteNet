import json
import os

FILE = "outputs/analytics.json"

def load_data():

    if not os.path.exists(FILE):

        data = {
            "glass":0,
            "metal":0,
            "plastic":0,
            "other":0
        }

        with open(FILE,"w") as f:
            json.dump(data,f)

    with open(FILE,"r") as f:
        return json.load(f)


def update_data(label):

    data = load_data()

    data[label] += 1

    with open(FILE,"w") as f:
        json.dump(data,f)


def get_data():

    return load_data()