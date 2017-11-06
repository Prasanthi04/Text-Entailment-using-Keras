# Text-Entailment-using-Keras
A sentence pair can be either entailing each other or contradicting each other or no relation called neutral. This code here classifies the SICK sentence pairs to one of the three classes using Deep Learning. Keras with LSTM  model is developed. Relatedness score is also calculated.

# How to run the code

The word_to_vector.py file is used to convert the bag of words of the corpus to n-dimensional vector space. The resulting output is the text file that will be called in the Entailment.py file for building the model.
The dd.py file is used to calculate the similarity score after preprocessing the text. Lowercase and POS tagging were used to do the task. dd.py is called in Entailment.py file to train the classifier. I am getting around 70% Accuracy with this model which is pretty good and there is always a scope for improvement

Relatedness score is calculated using the relatedness.py file, but the accuracy is only 62%, which has lot of room to improvel

# Contributions

Any suggestions are welcome!

# Contact
Contact me on prasanthi468@gmail.com if you have any trouble in executing this code. Happy coding :-)
