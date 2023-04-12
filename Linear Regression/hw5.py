import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def vis_data():
    data = pd.read_csv(sys.argv[1])
    plt.plot(data['year'], data['days'])
    plt.xlabel('Year')
    plt.ylabel('Number of frozen days')
    plt.savefig("plot.jpg")
    #plt.show()

def lin_reg():
    #Q3a: X
    df = pd.read_csv(sys.argv[1])
    df = pd.DataFrame(df)
    X = np.array((np.ones(len(df)), df['year']), dtype=np.int64).T

    print("Q3a:")
    print(X)

    #Q3b: Y
    Y = np.array(df["days"], dtype=np.int64)

    print("Q3b:")
    print(Y)

    #Q3c: matrix product
    Z = np.dot(X.T,X)

    print("Q3c:")
    print(Z)

    #Q3d: inverse
    I = np.linalg.inv(Z)

    print("Q3d:")
    print(I)

    #q3e: pseudo_inverse
    PI = np.dot(I,X.T)

    print("Q3e:")
    print(PI)

    #q3f: B^
    hat_beta = np.dot(PI,Y)

    print("Q3f:")
    print(hat_beta)

    #Q4
    x_test = 2022
    y_test = hat_beta[0] + hat_beta[1]*x_test

    print("Q4: " + str(y_test))

    #Q5:
    if hat_beta[1] > 0:
        print("Q5a: " + str(">"))
        print("Q5b: " + str("this means that the the number of frozen days on lake mendota is annually increasing on average according to the data"))
    elif hat_beta[1] < 0:
        print("Q5a: " + str("<"))
        print("Q5b: " + str("this means that the the number of frozen days on lake mendota is annually decreasing on average according to the data"))
    else:
        print("Q5a: " + str("="))
        print("Q5b: " + str("this means that the the number of frozen days on lake mendota is annually constant on average according to the data"))

    #Q6
    x_star = (0 - hat_beta[0]) / hat_beta[1]

    print("Q6a: " + str(x_star))
    print("Q6b: " + str("I do not think this x_star is a compelling prediction of the data because even though the data is decreasing over time, it does not appear that there would be a year with zero frozen days for a very long time, so based on these trends of the data, I do not think it is a compelling prediction."))

vis_data()
lin_reg()