import numpy as np
from sklearn.naive_bayes import BernoulliNB

# The task is to predict if the animal is a cow.

# Our binary features:
# - 0 Does it have four legs?
# - 1 Does it have fur?
# - 2 Is its color black and white?
# - 3 Is its color black only?
# - 4 Is its color white only?
# - 5 Is its color 'other'?
# - 6 Is it small(under 200kg)?

spider = [0, 0, 1, 0, 0, 0, 1]
cow_1 = [1, 1, 0, 0, 0, 1, 0]
cow_2 = [1, 1, 1, 0, 0, 0, 0]
dog_1 = [1, 1, 1, 0, 0, 0, 1]
dog_2 = [1, 1, 0, 1, 0, 0, 1]
cat_1 = [1, 1, 1, 0, 0, 0, 1]
snake = [0, 0, 0, 0, 0, 1, 1]
cow_3 = [1, 1, 1, 0, 0, 0, 0]
cow_4 = [1, 1, 0, 1, 0, 0, 0]
cat_2 = [1, 1, 0, 0, 1, 0, 1]
owl_1 = [0, 0, 1, 0, 0, 0, 1]
eleph = [1, 1, 0, 0, 0, 1, 0]
hippo = [1, 1, 0, 0, 0, 1, 0]
hhog_1 = [1, 1, 1, 0, 0, 0, 1]
hhog_2 = [1, 1, 0, 1, 0, 0, 1]
dog_3 = [1, 1, 0, 0, 0, 1, 1]
worm_1 = [0, 0, 0, 0, 0, 1, 1]
worm_2 = [0, 0, 0, 0, 0, 1, 1]
cow_5 = [1, 1, 0, 0, 1, 0, 0]

trainset = np.array([
    spider, cow_1, cow_2, dog_1, dog_2, cat_1, snake, cow_3, cow_4,
    cat_2, owl_1, eleph, hippo, hhog_1, hhog_2, dog_3, worm_1,
    worm_2, cow_5
])

is_cow = np.array([0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

classifier = BernoulliNB()
classifier.fit(trainset, is_cow)

# another white cow
test_0 = [1, 1, 0, 0, 1, 0, 0]
print("Is our new white cow a cow?")
print(classifier.predict([test_0]))

# a blue owl or a pink worm
test_1 = [0, 0, 0, 0, 0, 1, 1]
print("Is our blue owl a cow?")
print(classifier.predict([test_1]))

# black and white horse
test_2 = [1, 1, 1, 0, 0, 0, 0]
print("Is our black and white horse a cow?")
print(classifier.predict([test_2]))
