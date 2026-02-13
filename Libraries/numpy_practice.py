# 1-> array basics
#first creating arrays
import numpy as np
arr1=np.array([1,2,3,4])

#printing zeros and ones
zeros=np.zeros((2,3))
ones=np.ones((3,3))

#aranging and linspace
rangearray=np.arange(0,10,2) #like this will give elements within range with some diff
linspace=np.linspace(0,1,5)# this will divide the space btw into some eqal parts

#identity matrix
idt=np.eye(3) #this will give matrix with diagonals as 1

# array attributes
print("shape:", arr1.shape)
print("dimensions:", arr1.ndim)
print("size:", arr1.size)
print("ata Type:", arr1.dtype)

# 2->indexing slicing

arr = np.array([[10, 20, 30],
                [40, 50, 60],
                [70, 80, 90]])
# indexing
print(arr[1, 2]) # this will print element at particular idx

#slicing
print( arr[:2]) # this gives 1st two rows

# 3->math operations 
arr=np.array([1,2,3,4,5])
print(" addition:", arr + 5) # also called vectorization
print("multiplication:", arr * 2)
print("square:", arr ** 2)
# these all operations are performed as element wise
print("square root:", np.sqrt(arr))
print("log:", np.log(arr))
print("exponential:", np.exp(arr))

# 4-> broadcasting
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

vector = np.array([10, 20, 30])
ans=matrix+vector # this will ,here numpy considers shape of vector as (1,3)
column_mean = np.mean(matrix, axis=0)

# 5->reshaping
# reshape
reshaped = arr.reshape(3, 3) # this changes 1D-2D matrix
print( reshaped)
# flatten
flattened = reshaped.flatten()# this  reverses the changes
print( flattened)
#transpose
print( reshaped.T) # R->C
#concatenating
a = np.array([1, 2])
b = np.array([3, 4])
print( np.concatenate((a, b)))# joins array e-t-e

#*** like we can use arr.reshape(-1,3) -1 tells numpy auto the cal of req dim's

#6-> statistics
data = np.array([[85, 90, 88],
                 [78, 82, 84],
                 [92, 95, 91]])

print( np.mean(data))
print( np.mean(data, axis=1))
print( np.mean(data, axis=0))
print( np.std(data))
print( np.var(data))
print( np.percentile(data, 90))

#7-> linear algebra
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])
#matrix multiplication
print( np.dot(A, B))
# determinant
print( np.linalg.det(A)) # det of A is ad-bc
#Inverse
print( np.linalg.inv(A))
# eigenvalues
print( np.linalg.eigvals(A))

#-> random module
np.random.seed(42) # for not changing numbers everytime
print( np.random.randint(1, 10, (3, 3))) # this will generate the random rand num from 1-9 in 3*3 matrix

# Random floatmatrix
print( np.random.rand(2, 2))# gen ran num 0-1 in decimal

#normal distribution
# print(np.random.normal(0=>mean, 1=>sd, 5samples))
np.random.shuffle(X) # prevents model from learning oredr parameters
#******small exercise

import numpy as np

#students (rows) × subjects (columns)
marks = np.array([
    [85, 90, 88],
    [78, 82, 84],
    [92, 95, 91],
    [60, 65, 58]
])

#total-marks
total = np.sum(marks, axis=1)
print(total)

#avg marks
average = np.mean(marks, axis=1)
print(average)

#top student
top_student = np.argmax(total) # here argmax gives index and max gives value at index
print( top_student)

#failed students (avg< 60)
failed = np.where(average < 60)
print(failed)
