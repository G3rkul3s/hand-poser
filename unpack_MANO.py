import pickle
import numpy as np

with open('./MANO_RIGHT.pkl', 'rb') as f:
    data_right = pickle.load(f, encoding="latin1")

    v_template_right = data_right['v_template']
    shapedirs_right = data_right['shapedirs']
    shapedirs_right = np.transpose(shapedirs_right, (2, 0, 1))
    faces_right = data_right['f']
    joints_right = data_right['J']
    j_regressor_right = data_right['J_regressor'].toarray()
    weights_right = data_right['weights']

    np.savez("./MANO_RIGHT.npz",
            v_template=v_template_right,
            shapedirs=shapedirs_right,
            f=faces_right,
            J=joints_right,
            J_regressor=j_regressor_right,
            weights=weights_right)

with open('./MANO_LEFT.pkl', 'rb') as f:
    data_left = pickle.load(f, encoding="latin1")

    v_template_left = data_left['v_template']
    shapedirs_left = data_left['shapedirs']
    shapedirs_left_np = shapedirs_left.r.copy()
    shapedirs_left_np[:, 0, :] *= -1                            # fixes a shapedirs bug
    shapedirs_left = np.transpose(shapedirs_left_np, (2, 0, 1))
    faces_left = data_left['f']
    joints_left = data_left['J']
    j_regressor_left = data_left['J_regressor'].toarray()
    weights_left = data_left['weights']


    np.savez("./MANO_LEFT.npz",
            v_template=v_template_left,
            shapedirs=shapedirs_left,
            f=faces_left,
            J=joints_left,
            J_regressor=j_regressor_left,
            weights=weights_left)