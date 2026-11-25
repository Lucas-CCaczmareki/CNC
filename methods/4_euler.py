import numpy as np

def f(x, y):
    return -x*y

# exemplo do slide
# f, 0, 1, 1, 10
def euler(f, a, b, y0, m):

    if m < 1 or b <= a:
        print("m precisa ser >= 1 e b > a")
        return np.array([]), np.array([]), 1

    h = (b - a) / m
    x = a
    y = y0

    vet_x = np.zeros(m + 1)
    vet_y = np.zeros(m + 1)
    vet_x[0] = x
    vet_y[0] = y

    fxy = f(x, y)

    # até m+1 pq o python exclui o limite superior
    for i in range (1, m + 1):
        x = a + i * h
        y = y + h * fxy

        fxy = f(x, y)
        vet_x[i] = x # aqui fica i pq python é 0-based e o pseudoalg é 1-based
        vet_y[i] = y

    # return vet_x, vet_y, info(0 deu certo, 1 deu errado)
    return vet_x, vet_y, 0

# main
vet_x, vet_y, info = euler(f, 0, 1, 1, 10)
print(vet_x)
print()
print(vet_y)
print()
print((-vet_x)*vet_y)