from aiflow.lmeval import LLMEval

lme = LLMEval()

lme.setModel(
api=False, # means it will be used API instead of locally (your own)
url="http://localhost:5000/evaluate",
modelType="kobold"
)

lme.config(
temp=0.7,
window=2048,
Qamount=5 # five questions for test
)

results = lme.evaluate()
results.showResults()

# Save the results in a file (CSV, JSON, PNG)
resultsGraph = results.createGraph()
results.saveResults(resultsGraph, 'results.png')