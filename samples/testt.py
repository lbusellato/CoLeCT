import numpy as np
axis=np.array([-0.005137689143212892, -2.2130389147019818, 2.2250813729351666])
test = np.linalg.norm(axis)
print(test)
print(axis/test)
print(np.linalg.norm(axis/test))