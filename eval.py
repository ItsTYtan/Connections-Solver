import json
import os
from collections import Counter

directory = "results"

for filename in os.listdir(directory):
    with open(directory + "/" + filename, "r") as f:
        resultsData = json.load(f)

    totalCorrectGuesses = 0
    totalProblems = len(resultsData)
    for test in resultsData:
        guesses = test["guesses"]
        solutions = test["solution"]
        for guess in guesses:
            correctGuess = False
            for solution in solutions:
                if Counter(guess) == Counter(solution["members"]):
                    correctGuess = True

            if correctGuess:
                totalCorrectGuesses += 1

    print(filename + " avg correct guesses: " + str(totalCorrectGuesses/totalProblems))
    print(filename + " score: " + str(totalCorrectGuesses/(4 * totalProblems)))

    