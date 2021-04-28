import numpy as np

one_d = np.array([1.2, 2.4, 3.5])

print(one_d)  #one dimension array

two_d = np.array([1, 2], [2, 4], [5, 6])

print(two_d)  #two dimension array

#populate array with zeros

zero = np.zeros(4)  #[0,0,0,0]

#populate array with 1s

ones = np.ones(5)  #[1,1,1,1,1]
print(zero)
print(ones)

#poplaute with a sequence

sequence = np.arange(2, 20)
print(sequence)  #includes 2, but not 20

#populate with random integers

rand50to100 = np.random.randint(low=50, high=100)

print(rand50to100)  #is one less than the highest number

#random floting
ranfloat = np.random.random([4])
print(ranfloat)

#broadcasting to muliple vectors
