from __future__ import division
import numpy as np
from sklearn import hmm

states = ["illegal", "legal"]
n_states = len(states)

observations = ["hospital", "bar", "inn"]
n_observations = len(observations)

start_probability = np.array([0.6, 0.4])

transition_probability = np.array([
  [0.7, 0.3],
  [0.4, 0.6]
])

emission_probability = np.array([
  [0.1, 0.4, 0.5],
  [0.6, 0.3, 0.1]
])

model = hmm.MultinomialHMM(n_components=n_states)
model._set_startprob(start_probability)
model._set_transmat(transition_probability)
model._set_emissionprob(emission_probability)

# predict a sequence of hidden states based on visible states
ob = [0, 2, 1]
logprob, s = model.decode(ob, algorithm="viterbi")
print ", ".join(map(lambda x: observations[x], ob))
print ", ".join(map(lambda x: states[x], s))
print logprob