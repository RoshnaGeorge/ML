#qs
'''2. Write a program that takes a list of real numbers as input and returns the range (difference between minimum and maximum) of the list. Check for list being less than 3 elements in which case return an error message (Ex: “Range determination not possible”). Given a list [5,3,8,1,0,4], the range is 8 (8-0). '''
#algo
'''
input:enter the list of entries
function:keep a counter of 3,loop through the entries and if the counter turns 0,break out of the loop and continue,else return the error messg.Set the min and max to be the first entry of the list,then as we proceed if the next no is lesser,replace the min and more,replace the max.
return the output as a range
'''
#code
list1 = list(map(int, input("Enter elements of the list: ").split()))

def no_range(list1):
    if len(list1) < 3:
        print("Range determination not possible")
        return

    max1 = list1[0]
    min1 = list1[0]

    for i in range(1, len(list1)):
        if list1[i] > max1:
            max1 = list1[i]
        elif list1[i] < min1:
            min1 = list1[i]

    return max1 - min1

ans = no_range(list1)
print("Range is", ans)