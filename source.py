

import numpy as np
import pickle
import matplotlib.pyplot as plt
import numpy.linalg as lin
import scipy.linalg as la
import numpy.random as rnd
import math
import scipy.optimize as opt

"""
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
"""
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


def newtonsMethod(x, f, df, e):

    delta = dx(f,x)

    while math.fabs(f(x)) > e and math.fabs(delta) > e:

        x = x - f(x)/df(x)

        delta = dx(f, x)

        print "| x = "+str(x)+" | f(x) = " + str(f(x))+" | deltaX = "+str(delta)

""" UNCOMMENT THEM

newtonsMethod(2, f, df, 1e-11)

"""


print("---------------------------------QUESTION 6 b---------------------------------")

# b)


def fq6(x0, x1):

    a = math.pow(x0,2)+x0*x1 + math.pow(x1,2) - 3*x0 - x1 - 3

    b = 2*math.pow(x0,2) - x0*x1 - math.pow(x1,2) + x0 + 2*x1 - 1

    return np.array([a, b])


def df0x0(x0, x1):

    return 2*x0 - 3 + x1


def df0x1(x0, x1):

    return x0 + 2*x1 - 1


def df1x0(x0, x1):

    return 4*x0 - x1 + 1


def df1x1(x0, x1):

    return 2 - 2*x1 - x0


def J(x0, x1):

    return np.array([[df0x0(x0, x1), df0x1(x0, x1)], [df1x0(x0, x1), df1x1(x0, x1)]])


def q6b(x):

    counter = 0

    while counter < 10:

        s_k = np.negative(np.dot(lin.inv(J(x[0],x[1])), fq6(x[0],x[1])))

        x += s_k

        counter += 1

        print "x: "+str(x)+" ; norm of s: "+str(lin.norm(s_k))+"; norm of y: "+str(lin.norm(fq6(x[0],x[1])))


# c)


def mysolveQ6(A,b):

    P,L,U = la.lu(A)
    c = np.dot(np.transpose(P),b)
    y = la.solve_triangular(L,c,lower=True)
    x = la.solve_triangular(U,y,lower=False)
    return x


def q6c(x):

    counter = 0

    while counter < 10:

        s_k = mysolveQ6(J(x[0],x[1]), np.negative(fq6(x[0], x[1])))
        x += s_k
        counter += 1

        print "x: "+str(x)+" ; norm of s: "+str(lin.norm(s_k))+"; norm of y: "+str(lin.norm(fq6(x[0],x[1])))


"""UNCOMMENT

q6b([0,0])
print("---------------------------------QUESTION 6 c---------------------------------")
q6c([0,0])

"""


print("---------------------------------QUESTION 8 ---------------------------------")


def fq8(x0, x1):

    return 2*math.pow(x0, 4)+3*math.pow(x1, 4)-2*math.pow(x0, 2)*math.pow(x1, 2) - math.pow(x0, 2) - 4*math.pow(x1, 2) + 7

# a)


def gradient(x0, x1):

    a = 8*math.pow(x0, 3) - 2*math.pow(x1, 2)*x0 - 2*x0

    b = 12*math.pow(x1, 3) - 2*math.pow(x0, 2)*x1 - 8*x1

    return np.array([a, b])

# b)


def dfx0x0(x0, x1):

    return 24*x0*x0 - 2*x1*x1 - 2


def dfx0x1(x0, x1):

    return -4*x0*x1


def dfx1x1(x0, x1):

    return 36*math.pow(x1, 2) - 2*math.pow(x0, 2) - 8


def Hessian(x0, x1):

    return np.array([[dfx0x0(x0, x1), dfx0x1(x0, x1)], [dfx0x1(x0, x1), dfx1x1(x0, x1)]])


# c)

def q8c(x):

    counter = 0

    while counter < 10:

        s_k = np.negative(np.dot(lin.inv(Hessian(x[0], x[1])), gradient(x[0],x[1])))

        x += s_k

        counter += 1

        y = fq8(x[0], x[1])

        print "x: "+str(x)+" ; norm of s: "+str(lin.norm(s_k))+"; value of y: "+str(y)


# d)


def mysolveQ8(A,b):

    L = la.cholesky(A, lower=True)
    Lt = la.cholesky(A, lower=False)
    y = la.solve_triangular(L, b , lower=True)
    x = la.solve_triangular(Lt, y, lower=False)
    return x


def q8d(x):

    counter = 0

    while counter < 10:

        s_k = mysolveQ8(Hessian(x[0],x[1]), np.negative(gradient(x[0], x[1])))
        x += s_k
        counter += 1

        y = fq8(x[0], x[1])

        print "x: "+str(x)+" ; norm of s: "+str(lin.norm(s_k))+"; value of y: "+str(y)

'''
print("--------------------------------- QUESTION 8 c) ---------------------------------")
q8c([1, 1])
print("--------------------------------- QUESTION 8 d) ---------------------------------")
q8d([1, 1])
'''


# f)
"""
print("--------------------------------- QUESTION 8 f) ---------------------------------")

print("1st min:")
q8d([-1, -1])

print("2nd min:")
q8d([-1, 1])

print("3rd min")
q8d([1, 1])

print("4th min")
q8d([1, -1])

"""


print("--------------------------------- QUESTION 9 ---------------------------------")


mnist = np.load('npMnist.npy')

Xtrain = mnist


with open('tfMnist.pickle','rb'):
    new_mnist = pickle.load(f)

Xtest = new_mnist.test.images


c = []
k = []

# c)

def matfact(X, K):

    # Random matrix A

    counter = 0
    M, N = X.shape
    A = rnd.rand(M, K)

    while counter < 10:

        # Solve for Z:

        Z = lin.lstsq(A, X)[0]

        TSE1 = np.sum(np.power(np.subtract(X, np.dot(A,Z)),2))

        MSE1 = np.divide(TSE1, M*N)

        c.append(MSE1)

        print("MSE 1: "+str(MSE1))

        # Solve for A:
        A = np.transpose(lin.lstsq(np.transpose(Z), np.transpose(X))[0])

        TSE2 = np.sum(np.power(np.transpose(np.subtract(np.transpose(X), np.dot(np.transpose(Z),np.transpose(A)))),2))

        MSE2 = np.divide(TSE2,M*N)

        k.append(MSE2)

        print("MSE 2: "+str(MSE2))

        counter+=1

    return [np.dot(A, Z), A , Z]


#matfact(rnd.rand(784,700), 100)


def strictly_decreasing(L):
    return all(x>y for x, y in zip(L, L[1:]))


print(strictly_decreasing(c))
print(strictly_decreasing(k))


# d)
'''
matfact(Xtrain, 100)
'''
# e)

reconsImages = matfact(Xtrain, 100)[0]



def my_show_4_by_4(x, title):

    my_rand = np.arange(16)
    rnd.shuffle(my_rand)
    plt.figure()
    plt.suptitle(title)

    for i in range(1,17):
        plt.subplot(4,4,i)
        plt.imshow(np.reshape(x[i],[28,28]), cmap="Greys")
        plt.axis('off')
    plt.show()

'''
my_show_4_by_4(reconsImages, "Question 9(e): reconstructed images")
'''

# f)

# Random matrix A

Arand = rnd.rand(784,100)


# g)
'''
plt.hist(reconsImages, 100)
plt.show()
'''

# h)
"""
reconsImages[reconsImages < 0] = 0
reconsImages[reconsImages > 1] = 1
my_show_4_by_4(reconsImages, "Question 9(h): reconstructed images")
"""

# i)















