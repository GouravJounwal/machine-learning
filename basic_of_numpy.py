import numpy as np

a=np.array([[[1,2,3,9],[3,4,5,10],[6,7,8,10]],[[1,2,3,10],[3,4,5,10],
[6,7,8,10]],[[1,2,3,10],[3,4,5,10],[6,7,8,10]]])

print(a.ndim)

print(a.shape)

print(a.size)

a=np.arange(24).reshape(2,3,4)

a.resize(3,8)

print(a)
3
(3, 3, 4)
36
[[ 0  1  2  3  4  5  6  7]
 [ 8  9 10 11 12 13 14 15]
 [16 17 18 19 20 21 22 23]]
b = np.array([[1.5,2,3], [4,5,6]])
c = np.array( [ [1,2], [3,4] ], dtype=complex )

print(b)
print(b.dtype)

print(c)
[[1.5 2.  3. ]
 [4.  5.  6. ]]
float64
[[1.+0.j 2.+0.j]
 [3.+0.j 4.+0.j]]
c = np.array( [ [1,2], [3,4] ], dtype=complex )    # type of array explicitly defined
c
array([[1.+0.j, 2.+0.j],
       [3.+0.j, 4.+0.j]])
c=np.empty( (2,3) )
print(c)
b=np.ones( (2,3,4), dtype=np.int16 ) 
b
[[1.5 2.  3. ]
 [4.  5.  6. ]]
array([[[1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]],

       [[1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]]], dtype=int16)
b=np.ones( (2,3,4), dtype=np.int16 )
n=np.empty((2,5))
print(n)
[[4.64001887e-310 3.60739284e-313 1.38338381e-322 4.64001887e-310
  0.00000000e+000]
 [0.00000000e+000 4.94065646e-323 0.00000000e+000 0.00000000e+000
  0.00000000e+000]]
   
c=np.linspace( 0, 2, 9 )
ca=np.arange( 0, 2, 9 )
b=np.linspace( 0, 2, 5 )
ba=np.arange( 0, 2, 5 )
a=np.linspace( 0, 2, 20 )


aa=np.arange( 0, 2, 20 )



print(a ,aa ,b,ba,c,ca, sep='\n') 
[0.         0.10526316 0.21052632 0.31578947 0.42105263 0.52631579
 0.63157895 0.73684211 0.84210526 0.94736842 1.05263158 1.15789474
 1.26315789 1.36842105 1.47368421 1.57894737 1.68421053 1.78947368
 1.89473684 2.        ]
[0]
[0.  0.5 1.  1.5 2. ]
[0]
[0.   0.25 0.5  0.75 1.   1.25 1.5  1.75 2.  ]
[0]
import numpy as np 
a = np.array([0,30,45,60,90]) 

print('Sine of different angles:' )
# Convert to radians by multiplying with pi/180 
print(np.sin(a*np.pi/180) )
print('\n' ) 

print( 'Cosine values for angles in array:' )
print(np.cos(a*np.pi/180))
print ('\n' ) 

print('Tangent values for given angles:' )
print(np.tan(a*np.pi/180))
Sine of different angles:
[0.         0.5        0.70710678 0.8660254  1.        ]


Cosine values for angles in array:
[1.00000000e+00 8.66025404e-01 7.07106781e-01 5.00000000e-01
 6.12323400e-17]


Tangent values for given angles:
[0.00000000e+00 5.77350269e-01 1.00000000e+00 1.73205081e+00
 1.63312394e+16]
# Create an array with random values 
e = np.random.random((2, 2)) 
print ("\nA random array:\n", e) 
  
# Create a sequence of integers  
# from 0 to 30 with steps of 5 
f = np.arange(0, 30, 5) 
print ("\nA sequential array with steps of 5:\n", f) 
  
# Create a sequence of 10 values in range 0 to 5 
g = np.linspace(0, 5, 10) 
print ("\nA sequential array with 10 values between"
                                        "0 and 5:\n", g) 
  
# Reshaping 3X4 array to 2X2X3 array 
arr = np.array([[1, 2, 3, 4], 
                [5, 2, 4, 2], 
                [1, 2, 0, 1]]) 
  
newarr = arr.reshape(2, 2, 3) 
  
print ("\nOriginal array:\n", arr) 
print ("Reshaped array:\n", newarr) 
  
# Flatten array 
arr = np.array([[1, 2, 3], [4, 5, 6]]) 
flarr = arr.flatten() 
  
print ("\nOriginal array:\n", arr) 
print ("Fattened array:\n", flarr) 
A random array:
 [[0.61322028 0.48206154]
 [0.52683232 0.82510663]]

A sequential array with steps of 5:
 [ 0  5 10 15 20 25]

A sequential array with 10 values between0 and 5:
 [0.         0.55555556 1.11111111 1.66666667 2.22222222 2.77777778
 3.33333333 3.88888889 4.44444444 5.        ]

Original array:
 [[1 2 3 4]
 [5 2 4 2]
 [1 2 0 1]]
Reshaped array:
 [[[1 2 3]
  [4 5 2]]

 [[4 2 1]
  [2 0 1]]]

Original array:
 [[1 2 3]
 [4 5 6]]
Fattened array:
 [1 2 3 4 5 6]
Flatten array: We can use flatten method to get a copy of array collapsed into one dimension. It accepts order argument. Default value is ‘C’ (for row-major order). Use ‘F’ for column major order.

Stacking: Several arrays can be stacked together along different axes.

np.vstack: To stack arrays along vertical axis.

np.hstack: To stack arrays along horizontal axis.

np.column_stack: To stack 1-D arrays as columns into 2-D arrays. np.concatenate: To stack arrays along specified axis (axis is passed as argument).

import numpy as np
  
a = np.array([[1, 2], 
              [3, 4]]) 
  
b = np.array([[5, 6], 
              [7, 8]]) 
  
# vertical stacking 
print("Vertical stacking:\n", np.vstack((a, b))) 
  
# horizontal stacking 
print("\nHorizontal stacking:\n", np.hstack((a, b))) 
  
c = [5, 6] 
  
# stacking columns 
print("\nColumn stacking:\n", np.column_stack((a, c))) 
  
# concatenation method  
print("\nConcatenating to 2nd axis:\n", np.concatenate((a, b), 1)) 
Vertical stacking:
 [[1 2]
 [3 4]
 [5 6]
 [7 8]]

Horizontal stacking:
 [[1 2 5 6]
 [3 4 7 8]]

Column stacking:
 [[1 2 5]
 [3 4 6]]

Concatenating to 2nd axis:
 [[1 2 5 6]
 [3 4 7 8]]
from google.colab import drive
drive.mount('/content/drive')
Mounted at /content/drive
Splitting:For splitting, we have these functions: np.hsplit: Split array along horizontal axis. np.vsplit: Split array along vertical axis. np.array_split: Split array along specified axis.

import numpy as np 
  
a = np.array([[1, 3, 5, 7, 9, 11], 
              [2, 4, 6, 8, 10, 12]]) 
  
# horizontal splitting 
print("Splitting along horizontal axis into 2 parts:\n", np.hsplit(a, 2)) 
  
# vertical splitting 
print("\nSplitting along vertical axis into 2 parts:\n", np.vsplit(a, 2))
Splitting along horizontal axis into 2 parts:
 [array([[1, 3, 5],
       [2, 4, 6]]), array([[ 7,  9, 11],
       [ 8, 10, 12]])]

Splitting along vertical axis into 2 parts:
 [array([[ 1,  3,  5,  7,  9, 11]]), array([[ 2,  4,  6,  8, 10, 12]])]
# broadcasting Strenching array
import numpy as np 
  
a = np.array([0.0, 10.0, 20.0, 30.0])

b = np.array([0.0, 1.0, 2.0,5.0]) 

  
print(a + b)
[ 0. 11. 22. 35.]
a = np.ones(3, dtype=np.int32)
b = np.linspace(0, 3, 4)
print(a)
print(b)
print(a.sum(), b.min(), b.max())
[1 1 1]
[0. 1. 2. 3.]
3 0.0 3.0
