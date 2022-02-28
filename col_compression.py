import numpy as np
import os
def to_column_1(array, temp_path, zeros= True):
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    for i in range(array.shape[0]):
        for j in range(array.shape[0]):
            if zeros:
                if array[i, j].n == 0:
                    continue
            provenance = array[i, j].prov
            for id, x, y in provenance:
                x1.append(i)
                x2.append(j)
                y1.append(x)
                y2.append(y)

    x1 = np.array(x1)
    path = os.path.join(temp_path, 'x1.npy')
    np.save(path, x1)

    x2 = np.array(x2)
    path = os.path.join(temp_path, 'x2.npy')
    np.save(path, x2)

    y1 = np.array(y1)
    path = os.path.join(temp_path, 'y1.npy')
    np.save(path, y1)

    y2 = np.array(y2)
    path = os.path.join(temp_path, 'y2.npy')
    np.save(path, y2)

def to_column_2(array, temp_path, input_id = (1, 2), zeros= True):
    x1 = []
    x2 = []
    y1 = []
    y2 = []

    w1 = []
    w2 = []
    z1 = []
    z2 = []
    for i in range(array.shape[0]):
        for j in range(array.shape[0]):
            if zeros:
                if array[i, j].n == 0:
                    continue
            provenance = array[i, j].prov
            for id, x, y in provenance:
                if id == input_id[0]:
                    x1.append(i)
                    x2.append(j)
                    y1.append(x)
                    y2.append(y)
                else:
                    w1.append(i)
                    w2.append(j)
                    z1.append(x)
                    z2.append(y)
                    z1.append(-1)
                    z2.append(-1)

    x1 = np.array(x1)
    path = os.path.join(temp_path, 'x1.npy')
    np.save(path, x1)

    x2 = np.array(x2)
    path = os.path.join(temp_path, 'x2.npy')
    np.save(path, x2)

    y1 = np.array(y1)
    path = os.path.join(temp_path, 'y1.npy')
    np.save(path, y1)


    y2 = np.array(y2)
    path = os.path.join(temp_path, 'y2.npy')
    np.save(path, y2)

    w1 = np.array(w1)
    path = os.path.join(temp_path, 'w1.npy')
    np.save(path, w1)

    w2 = np.array(w2)
    path = os.path.join(temp_path, 'w2.npy')
    np.save(path, w2)

    z1 = np.array(z1)
    path = os.path.join(temp_path, 'z1.npy')
    np.save(path, z1)

    z2 = np.array(z2)
    path = os.path.join(temp_path, 'z2.npy')
    np.save(path, z2)