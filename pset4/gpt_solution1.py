'''
LIGN 167 PSET 4
Kyra Fetter, A16819058
Lina Battikha, A16852105
Oishani Bandopadhyay, A16800427

Group Member Contributions:
Kyra: 9, 10, 11
Lina: 1, 2, 3, 4
Oishani: 5, 6, 7, 8 
Everyone: Reviewing code and debugging
'''

# Problem 1 - Lina


# Problem 2 - Lina


# Problem 3 - Lina


# Problem 4 - Lina


# Problem 5 - Oishani


# Problem 6 - Oishani


# Problem 7 - Oishani


# Problem 8 - Oishani


# Problem 9 - Kyra
'''
Link: https://chat.openai.com/share/751ea31a-21ca-446f-bccc-6e851fb603e9
GPT-4's response is correct. It accurately identifies the relationship between 
decreasing loss and increasing log-likelihood and thus sharper probability 
distribution. 

When the train function is called, the Elman RNN is trained over a series of epochs.
The output you provided shows the loss at the end of each epoch. The loss is a
measure of how well the RNN is performing—the lower the loss, the closer the RNN's
output is to the expected result.

Over time, as seen from the output, the loss is decreasing. This decrease in loss
implies that the RNN is learning from the data. Specifically, it suggests that
the RNN's predictions are getting more accurate, and thus the model is improving
its estimation of the probability distribution of the next word given the previous
context (previous words in the sentence).

In language modeling, the loss typically used is the negative log-likelihood of
the correct word given the context. As the model trains and the loss decreases,
it means that the negative log likelihood is getting smaller, which in turn means
that the likelihood (or probability) of the correct word is increasing. Therefore,
the probability distribution learned by the model is becoming sharper and more
focused on the correct outcomes, indicating that the model's predictions are
becoming more confident and accurate over time.

In practical terms, this learned probability distribution can be used to generate
text, where the model would pick words with the highest probability as the next
word in a sequence, or it can be used to assign probabilities to existing sentences,
which is useful in tasks like speech recognition, machine translation, and other
natural language processing applications.
'''


# Problem 10 - Kyra


# Problem 11 - Kyra


