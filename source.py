

import numpy as np
import pickle
import matplotlib.pyplot as plt
import numpy.linalg as lin
import scipy.linalg as la
import numpy.random as rnd
import math
import scipy.optimize as opt


print("---------------------------------QUESTION 1---------------------------------")


# a)

a = np.array([3,1,2,4])

alpha = lin.norm(a)

print("alpha = " + str(alpha))

# To avoid cancellation error

alpha = -alpha

v = np.subtract(a, np.array([alpha, 0, 0, 0]))

print("v = "+str(v))


v_t_a = np.dot(np.transpose(v),a)

print(v_t_a)

v_t_v = np.dot(np.transpose(v),v)

print(v_t_v)

b = np.dot(2, np.divide(v_t_a,v_t_v))

print("2*((v^t*a)/(v^t*v)) = "+str(b))

# the value of b is 1, no need to multiply b and v

Ha = np.subtract(a, v)

print("Ha = " + str(Ha))


H = np.subtract(np.identity(4), np.dot(2,np.divide(np.multiply(v,np.transpose(v)), np.multiply(np.transpose(v), v))))

print(H)

print("---------------------------------QUESTION 3---------------------------------")

# a)

def plot_q3(my_range, title, left, right, leftLabel, rightLabel):

    fig = plt.figure()
    fig.suptitle("Question 3. "+str(title))
    plt.plot(my_range, left,  color='blue', label = leftLabel)
    plt.plot(my_range, right,  color='orange', label = rightLabel)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def left_q3_a(x):

    a = np.subtract(np.power(x,3) , np.multiply(5,np.power(x,2)))

    b = np.subtract(np.multiply(8,x), 4)

    return np.add(a,b)


def right_q3_a(x):
    return 0*x


# b)

def left_q3_b(x):

    return np.multiply(x, np.cos(np.multiply(20,x)))


def right_q3_b(x):

    return np.power(x,3)

# c)


def left_q3_c(x):

    a = np.exp(np.negative(np.dot(2,x)))

    b = np.exp(x)

    return np.add(a,b)


def right_q3_c(x):

    return np.add(x,4)


my_range = np.linspace(0, 3, 1000)

# Assigning a)
left_q3_a = left_q3_a(my_range)
right_q3_a = right_q3_a(my_range)

# Assigning b)

left_q3_b = left_q3_b(my_range)
right_q3_b = right_q3_b(my_range)

# Assigning c)

left_q3_c = left_q3_c(my_range)
right_q3_c = right_q3_c(my_range)

# Plotting a,b,c

""" UNCOMMENT THEM
plot_q3(my_range, " a) ", left_q3_a, right_q3_a, "x^3-5x^3+8x-4", "0")

plot_q3(my_range, " b) ", left_q3_b, right_q3_b, "xcos(20x)", "x^3")

plot_q3(my_range, " c) ", left_q3_c, right_q3_c, "e^(-2x)+e^x", "x+4")
"""


print("---------------------------------QUESTION 4---------------------------------")



def f(x):

    return math.exp(-2*x) + math.exp(x) - x - 4


def df(x):

    return math.exp(x) - 2*math.exp(-2*x) - 1

'''
root = opt.newton(f, 2 , fprime=df, tol=1e-11)

print("root is " +str(root))
'''


def dx(f, x):
    return math.fabs(0-f(x))


def newtonsMethod(x, f, df , e ):

    delta = dx(f,x)

    while math.fabs(f(x)) > e and math.fabs(delta) > e:

        x = x - f(x)/df(x)

        delta = dx(f, x)

        print "| x = "+str(x)+" | f(x) = " + str(f(x))+" | deltaX = "+str(delta)


newtonsMethod(2, f , df, 1e-11)




