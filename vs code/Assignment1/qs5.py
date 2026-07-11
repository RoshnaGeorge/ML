#qs
'''5. Generate a list of 25 random numbers between 1 and 10. Find the mean, median and mode for these numbers.'''
#algo
'''
generate 25 random no between 1 to 10
function:find the mean,median and mode for the 25 rand no using the import statistics module.
'''
#code
import random,statistics
num = []
for i in range(25):
    num.append(random.randint(1, 10))
print("Array:", num)
mean = statistics.mean(num)
median = statistics.median(num)
mode = statistics.mode(num)
print("Mean:", mean)
print("Median:", median)
print("Mode:", mode)