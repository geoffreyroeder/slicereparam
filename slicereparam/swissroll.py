#%%
import numpy as np
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt

def get_swiss_roll_data(filename, N, noise, scale):
    try:
        print("Trying to load the data from file.")
        X = np.loadz(filename)
        # check that there are N samples
        if X.shape[0] != N:
            raise Exception("Number of samples does not match, recomputing.")
        print("Data loaded successfully.")
    except Exception:
        print("Failed to load data; generating new Swiss roll data.")
        X, _ = make_swiss_roll(N, noise=noise)
        X /= scale
        # save
        np.savez(filename, X=X)
        print("Data generated and saved.")
    return X

#%%
if __name__ == "__main__":
    N = 20_000
    noise = 0.2
    scale = 1
    filename = f"swiss_roll_{N}_{noise}_{scale}.npz"
    data = get_swiss_roll_data(filename, N, noise, scale)
    plt.figure(figsize=(8, 6))
    # 3D scatter plot
    ax = plt.axes(projection='3d')
    ax.scatter3D(data[:,0], data[:,1], data[:,2], c=data[:,0], cmap=plt.cm.Spectral, alpha=0.4)
    plt.title("Visualization of Swiss Roll Data")
    # label axes as X, Y, Z
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.savefig(f"swiss_roll_{N}_{noise}_{scale}.pdf")

# %%
