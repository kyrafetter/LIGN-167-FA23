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
When the code is run by calling train, the loss decreases over time, indicating an
increase in log-liklihood. The loss initially was approximately 7.316, but then
exponentially decreased until it finally converged at 5.019617 after 100 epochs.
Generally speaking, because the loss decreased over training, we know that the
probability associated with the next word prediction was increasing overtime.
The loss initially decreased with greater steps, demonstrating that predictions
with high-error were made towards the beginning of training; however, as the
magnitude of the change in loss decreased throughout training, we see that the model's 
probability distribution around correct words became increasingly sharper and flatter
elsewhere as fewer and fewer high-error predictions were made.
'''


# Problem 10 - Kyra


# Problem 11 - Kyra
'''
The Elman network would not be able to model this dependency completely, if at all.
This is because the interpretation of the 950th word is determined solely from that
of the 949th word, which is representative of itself and all words before it. By this
point, we have calculated many many gradients such that we see the exploding/vanishing
gradient problem, where the effect of words at the beginning of the sentence (for
ex. the 3rd word) disappear and are lost due to the many multiplications of what may
become very large or very small values. We also may run into the issue of the 
accumuation of noise, especially if the training text used is long. 
'''