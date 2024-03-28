import json
import matplotlib.pyplot as plt
import requests

class LLMEval:
    def __init__(self):
        """
        Initialize the LLMEval object. The model and evaluation parameters must be set before calling the evaluate method.
        """

        self.api_mode = False
        self.url = None
        self.modelType = None
        self.temp = None
        self.window = None
        self.Qamount = None
        self.results = []

    def setModel(self, api=False, url=None, modelType=None):
        """
        Set the model to be evaluated. If the API mode is enabled, the URL of the API must be provided.
        """

        self.api_mode = api
        self.url = url
        self.type = modelType

    def config(self, temp=0.7, window=2048, Qamount=5):
        """
        Configure the evaluation parameters. The default values are: temp=0.7, window=2048, Qamount=5
        """

        self.temp = temp
        self.window = window
        self.Qamount = Qamount

    def evaluate(self):
        """
        Start the evaluation process. If the API mode is enabled, it will send a request to the API to evaluate the model. Otherwise, it will evaluate the model locally.
        """

        allowed_models = ['kobold', 'gpt-3', 'bert', 'gpt-2']
    
        if self.api_mode and self.url and self.modelType in allowed_models:
            response = requests.post(self.url, json={
                'temp': self.temp,
                'window': self.window,
                'Qamount': self.Qamount
            })
            self.results = response.json()
        else:
            # Evaluate the model locally and populate self.results
            # with the scores for each question
            self.results = [80, 20, 100, 60, 40]  # Mock results
        return self

    def showResults(self):
        """
        Display the evaluation results in the console. If there are no results, it will print a message indicating that there are no results to show.
        """

        if not self.results:
            print("No results to show.")
            return
        total_score = 0
        print("Results:")

        for i, score in enumerate(self.results, start=1):
            print(f"> Question {i}: {score}%")
            total_score += score
        average_score = total_score / len(self.results)
        print(f"\n> Average: {average_score:.2f}% / 100%")

    def createGraph(self):
        """
        Create a bar graph with the evaluation results. If there are no results, it will print a message indicating that there are no results to create a graph.
        """

        if not self.results:
            print("No results to create a graph.")
            return None

        plt.figure(figsize=(10, 5))
        questions = [f"Q{i+1}" for i in range(len(self.results))]
        plt.bar(questions, self.results, color='skyblue')
        plt.xlabel('Questions')
        plt.ylabel('Scores (%)')
        plt.title('Evaluation Results')
        plt.ylim(0, 100)
        plt.axhline(y=sum(self.results) / len(self.results), color='r', linestyle='--')
        plt.text(len(self.results)-1, sum(self.results) / len(self.results) + 2, f"Average: {sum(self.results) / len(self.results):.2f}%")
        return plt

    def saveResults(self, graph, filename):
        """
        Save the evaluation results in a file. The file format is determined by the file extension (e.g., .csv, .json, .png).
        """

        if graph:
            graph.savefig(filename)
            print(f"Results saved as {filename}.")
        else:
            print("No graph to save.")