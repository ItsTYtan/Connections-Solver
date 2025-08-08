import json
import os
import random
from datasets import Dataset
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
login(os.getenv("HUGGINGFACE_TOKEN"))

problems = []
solutions = []

# load archive
with open("NYT-Connections-Answers/connections.json", "r") as archive:
    data = json.load(archive)
    for puzzle in data:
        answers = puzzle["answers"]

        words = []
        test_answers = []
        for answer in answers:
            words.extend(answer["members"])

            del answer["level"]
            test_answers.append(answer)

        # Shuffle words to make it random
        random.seed(42)
        random.shuffle(words)

        problems.append(words)
        solutions.append(test_answers)

unformatted_dataset = []

for problem, solution in list(zip(problems, solutions)):
    unformatted_dataset.append(
        {
            "puzzle": problem,
            "solution": solution
        }
    )

dataset = Dataset.from_list(unformatted_dataset)
split_dataset = dataset.train_test_split(test_size=0.1)

split_dataset.push_to_hub("ItsTYtan/nytimes-connections")