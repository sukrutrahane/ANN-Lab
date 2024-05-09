def mcullochpitts(x,w,treshold):
    net_input= sum([x[i]*w[i] for i in range(len(x))])
    if net_input >=treshold:
        return 1
    else:
        return 0

def andnot(x1,x2):
    x=[x1,x2]
    w=[w1,w2]
    treshold=2
    output= mcullochpitts(x,w, treshold)
    return output

w1=int(input("Enter weight for X1: "))
w2=int(input("Enter weight for X2: "))
print("X1: 0, X2: 0 ---> ANDNOT: ", andnot(0,0))
print("X1: 0, X2: 1 ---> ANDNOT: ", andnot(0,1))
print("X1: 1, X2: 0 ---> ANDNOT: ", andnot(1,0))
print("X1: 1, X2: 1 ---> ANDNOT: ", andnot(1,1))
