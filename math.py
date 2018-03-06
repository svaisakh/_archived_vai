import numpy as np

spherical_range_tensor = np.zeros((3, 2))
spherical_range_tensor[0, 0] = np.sqrt(3)
spherical_range_tensor[0, 1] = np.sqrt(3) * 256
spherical_range_tensor[1, 0] = np.arccos(256 / np.sqrt(256 ** 2 + 2))
spherical_range_tensor[1, 1] = np.arccos(1 / np.sqrt(2 * 256 ** 2 + 1))
spherical_range_tensor[2, 1] = np.arctan(256)
spherical_range_tensor[2, 0] = np.pi / 2 - spherical_range_tensor[2, 1]

def to_spherical(cartesian_tensor):
    cartesian_tensor = cartesian_tensor.astype(float) + 1
    
    x, y, z = tuple(np.take(cartesian_tensor, i, -1) for i in range(3))

    r = np.linalg.norm(cartesian_tensor, axis=-1)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    
    spherical_tensor = np.stack([r, theta, phi], -1)
    
    spherical_tensor = (spherical_tensor - spherical_range_tensor[:, 0]) * 255 / (spherical_range_tensor[:, 1] - spherical_range_tensor[:, 0])
    return spherical_tensor

def to_cartesian(spherical_tensor):
    spherical_tensor = spherical_tensor.astype(float) * (spherical_range_tensor[:, 1] - spherical_range_tensor[:, 0]) / 255 + spherical_range_tensor[:, 0]
    
    r, theta, phi = tuple(np.take(spherical_tensor, i, -1) for i in range(3))

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    cartesian_tensor = np.stack([x, y, z], -1)
    
    return cartesian_tensor - 1
