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
GPT-4 stood by its first response and did not provide any significant changes.
This new response remains correct.

Let's review the response and identify the error.

Upon revisiting the explanation, I see that the description of the loss and its
implications for the probability distribution learned by the model was accurate.
The loss is indeed decreasing over time, which suggests the RNN is learning to
better predict the next word in a sequence based on the context provided by the
preceding words.

However, I should clarify the relationship between the loss and the probability
distribution: a decreasing loss over training epochs implies that the RNN's
estimated probability distribution over sequences is increasingly aligning with
the true distribution of the training data. The model is not just becoming more
confident, but more importantly, it is becoming more correct in its predictions.

The error may not lie in the conceptual explanation but could be a misunderstanding
of expectations. If the error is expected to be in the code description, please
provide more information, and I'll address it accordingly. If the error is expected
to be in the interpretation of the training output, then the explanation above
stands correct: a decreasing loss is typically a sign of a model learning and
improving its performance on the training data.
'''


# Problem 10 - Kyra


# Problem 11 - Kyra
'''
Link: https://chat.openai.com/share/09f4bea1-e8e8-4300-97bc-9f1655889ecb
GPT-4 provided essentially the same response, but included mention of the Exploding
Gradient Problem, which it did not explicity identify in it's first response. However,
this is a valid problem, and therefore GPT-4's response remains correct.

Apologies for the confusion earlier. Let's reevaluate the situation described 
in Problem 11.

Elman networks are a type of recurrent neural network that can theoretically 
maintain information from previous time steps through hidden states. However, 
they are limited in their capacity to handle long-range dependencies, especially 
over thousands of words as mentioned in the scenario.

Here are the primary challenges they would face:

Vanishing Gradient Problem: In long sequences, as the gradient of the loss is 
propagated backward through each timestep, it can shrink exponentially, becoming 
vanishingly small. This makes it difficult for the network to learn dependencies 
between distant positions in the sequence.

Limited Capacity of Hidden States: The hidden state in an Elman network carries 
information from one timestep to the next. As the sequence progresses, earlier 
information tends to be overwritten by more recent information. There's a limit 
to how much information the hidden state can carry, and over a long sequence like 
1000 words, it's unlikely that the information from the 3rd word would be preserved 
all the way to the 950th word.

Exploding Gradient Problem: Conversely to vanishing gradients, gradients can also 
grow exponentially during backpropagation, which can lead to very large updates to 
the model weights and cause the model to diverge.

Given these issues, an Elman network might struggle to capture the dependency of 
the 950th word on the 3rd word in a text. Advanced architectures like LSTMs or 
GRUs are better suited for this task as they have mechanisms to remember and 
forget information selectively over long sequences, helping to preserve relevant 
information and mitigate the vanishing gradient problem.
'''