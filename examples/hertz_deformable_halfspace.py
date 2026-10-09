"""Qualified axisymmetric deformable-half-space Hertz benchmark."""
from tensorfem.hertz_axisymmetric import hertz_reference,solve_axisymmetric_hertz

reference=hertz_reference(.025,10.,1e5,.3)
print("mesh  load error(%)  radius error(%)  pressure error(%)")
for n in (16,24,40):
    result=solve_axisymmetric_hertz(nr=n,nz=n)
    errors=[100*abs(float(value)/exact-1) for value,exact in
            zip((result.load,result.contact_radius,result.peak_pressure),reference)]
    print(f"{n:4d} {float(result.load):10.4f} {errors[0]:10.4f} "
          f"{errors[1]:15.4f} {errors[2]:17.4f}")

