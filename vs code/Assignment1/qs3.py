#qs
'''3. Write a program that accepts a square matrix A and a positive integer m as arguments and 
returns A^m. 
'''
#algo
'''
input:enter the matrix entries and positive int m
function:binary exponentiation:so if the power is even,then split it recursively in the terms of 2 X m/2,then solve it in the recursive function,if the power is odd then multiply the matrix once with the resultant matrix after recursive call for the even power.
result is the output of the multiplication.
'''
#code
n = int(input("Enter the order: "))
A = []
print("Enter the matrix:")
for i in range(n):
    row = list(map(int, input().split()))
    A.append(row)
pow1 = int(input("Enter the power: "))

#function to only multiply 2 matrices
def mul1(A, B):
    n = len(A)
    res = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(0)
        res.append(row)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                res[i][j] += A[i][k] * B[k][j]
    return res

#function to classify the powers-0-identity,1-self,even,odd
def matrix_pow(A, pow1):
    if pow1 == 0:#identity matrix
        n = len(A)
        iden = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(1)
                else:
                    row.append(0)
            iden.append(row)
        return iden
    if pow1 == 1:
        return A
    if pow1 % 2 == 0:#even power
        temp = matrix_pow(A, pow1 // 2)
        return mul1(temp, temp)
    else:#odd power
        temp = matrix_pow(A, (pow1 - 1) // 2)
        temp = mul1(temp, temp)
        return mul1(temp, A)


answer = matrix_pow(A, pow1)
for row in answer:
    for element in row:
        print(element, end=" ")
    print()

