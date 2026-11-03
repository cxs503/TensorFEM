import sympy as sp
xi = sp.symbols('xi', positive=True)
A = sp.Matrix([[0,1,0,0],[0,0,1,0],[0,0,0,1],[-4,0,0,0]])
P = sp.exp(A*xi)  # sympy exact matrix exponential
P = sp.simplify(P)
for i in range(4):
    print([sp.simplify(sp.expand(sp.powsimp(P[i,j], force=True))) for j in range(4)])
print("--- rewrite with e^-xi ---")
for i in range(4):
    row=[]
    for j in range(4):
        row.append(sp.simplify(P[i,j].rewrite(sp.exp).expand()))
    print(row)
