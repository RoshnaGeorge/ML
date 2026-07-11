#qs
'''Consider the given list as [2, 7, 4, 1, 3, 6]. Write a program to count pairs of elements with 
sum equal to 10. 
'''
#algo
'''
input:enter the list entries.
function:take a number,find if there is any other number in the list who’s value is equal to 10-chosen no.If yes,increase the count,else go to next ele.
print the final output
'''
#code
list1 = list(map(int, input("Enter elements of the list: ").split()))

def count_pairs(list1):
    count = 0

    for i in range(len(list1)):
        for j in range(i + 1, len(list1)):
            if list1[i] + list1[j] == 10:
                count += 1

    return count

ans = count_pairs(list1)
print(f"The total no of element pairs adding upto 10 is {ans}")
            
