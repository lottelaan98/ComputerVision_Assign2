import cv2
import numpy as np

def load_camera_parameters(xml_path):
    fs = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)

    K = fs.getNode("camera_matrix").mat()
    d = fs.getNode("distortion_coefficients").mat()
    r = fs.getNode("rotation_matrix").mat()
    t = fs.getNode("translation_vector").mat()

    fs.release()
    return K, d, r, t


def project_voxels(voxel_points, rvec, tvec, K, dist):
    imgpts, _ = cv2.projectPoints(
        voxel_points,   # Nx3
        rvec,
        tvec,
        K,
        dist
    )
    return imgpts



x_range = np.arange(-1.0, 1.0, 0.02)
y_range = np.arange(-1.0, 1.0, 0.02)
z_range = np.arange(0.0, 2.0, 0.02)
voxels = np.array(np.meshgrid(x_range, y_range, z_range)).T.reshape(-1,3)

if __name__ == "__main__":

    K, d, r, t = load_camera_parameters("data/cam1/config.xml")

    projected_points = project_voxels(
        voxel_points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
        rvec=cv2.Rodrigues(r)[0],
        tvec=t,
        K=K,
        dist=d
    )

    print("Projected voxel points:\n", projected_points.reshape(-1, 2))

    print("Camera matrix:\n", K)
    print("Distortion:\n", d)
    print("Rotation:\n", r)
    print("Translation:\n", t)
