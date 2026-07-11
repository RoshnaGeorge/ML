#qs
'''4. Write a program to count the highest occurring character & its occurrence count in an input string. Consider only alphabets. Ex:  for “hippopotamus” as input string, the maximally occurring character is ‘p’ & occurrence count is 3. '''
#algo
'''
input:enter a string as input
function:go letter by letter,first letter make it dictionary with count 1.then with every iteration check if the key alphabet already exists in the dict,if not make one else update the dict.After that find the highest/max value and print the key and value.
return the final key:value as ans.
'''
#code
word = input("Enter a word: ")

def letter_count(word):

    letters = {}
    for i in word:
        if i in letters:
            letters[i] += 1
        else:
            letters[i] = 1
    return letters

def max_count(letters):
    max_count = 0

    for value in letters.values():
        if value > max_count:
            max_count = value

    return max_count

dict_letters=letter_count(word)
max_value=max_count(dict_letters)

for key in dict_letters:
    if dict_letters[key] == max_value:
        print(key, dict_letters[key])

