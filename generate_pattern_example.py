import generate_pattern as gp

"""#demonstrate on generate hadamard mask, generate 128 * 128 within 15s.
DP = gp.DmdPattern("hadamard", 128, 128)
# pattern is the mask point oneside while the conjugate_pattern is the pattern point other side.
pattern, conjugate_pattern = DP.execute()
print(pattern, conjugate_pattern)
"""
DP = gp.DmdPattern("random", 128, 128)
for i in range(128*128):
    pattern, conjugate_pattern = DP.execute(random_sparsity=0.5)#fifty percent elements in array is 1
