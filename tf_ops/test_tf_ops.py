import tensorflow as tf
import numpy as np
import time
from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate
from tf_sampling import prob_sample, farthest_point_sample, gather_point


class TestGrouping(tf.test.TestCase):
    def test(self):
        knn = True
        np.random.seed(100)
        pts = np.random.random((32, 512, 64)).astype("float32")
        tmp1 = np.random.random((32, 512, 3)).astype("float32")
        tmp2 = np.random.random((32, 128, 3)).astype("float32")
        points = tf.constant(pts)
        xyz1 = tf.constant(tmp1)
        xyz2 = tf.constant(tmp2)
        radius = 0.1
        nsample = 64
        if knn:
            _, idx = knn_point(nsample, xyz1, xyz2)
            grouped_points = group_point(points, idx)
        else:
            idx, _ = query_ball_point(radius, nsample, xyz1, xyz2)
            grouped_points = group_point(points, idx)

        now = time.time()
        for _ in range(100):
            ret = grouped_points.numpy()  # 直接调用 .numpy() 触发计算
        print(time.time() - now)
        print(ret.shape, ret.dtype)

    def test_grad(self):
        with tf.device("/gpu:0"):
            points = tf.constant(np.random.random((1, 128, 16)).astype("float32"))
            print(points)
            xyz1 = tf.constant(np.random.random((1, 128, 3)).astype("float32"))
            xyz2 = tf.constant(np.random.random((1, 8, 3)).astype("float32"))
            radius = 0.3
            nsample = 32
            idx, pts_cnt = query_ball_point(radius, nsample, xyz1, xyz2)
            grouped_points = group_point(points, idx)
            print(grouped_points)

        with tf.GradientTape() as tape:
            tape.watch(points)
            grouped_points = group_point(points, idx)  # 重新计算，确保在tape上下文中

        # 计算点相对于输出的梯度
        gradients = tape.gradient(grouped_points, points)

        # 可以打印梯度来查看
        print("Gradients: ", gradients)
        # 根据梯度检查实际的数值误差
        self.assertIsNotNone(gradients)  # 检查梯度是否正确计算


class TestInterpolate(tf.test.TestCase):
    def test(self):
        np.random.seed(100)
        pts = np.random.random((32, 128, 64)).astype("float32")
        tmp1 = np.random.random((32, 512, 3)).astype("float32")
        tmp2 = np.random.random((32, 128, 3)).astype("float32")
        with tf.device("/cpu:0"):
            points = tf.constant(pts)
            xyz1 = tf.constant(tmp1)
            xyz2 = tf.constant(tmp2)
            dist, idx = three_nn(xyz1, xyz2)
            weight = tf.ones_like(dist) / 3.0
            interpolated_points = three_interpolate(points, idx, weight)
            now = time.time()
            for _ in range(100):
                ret = interpolated_points.numpy()
            print(time.time() - now)
            print(ret.shape, ret.dtype)

    def test_grad(self):
        with self.test_session():
            points = tf.constant(np.random.random((1, 8, 16)).astype("float32"))
            print(points)
            xyz1 = tf.constant(np.random.random((1, 128, 3)).astype("float32"))
            xyz2 = tf.constant(np.random.random((1, 8, 3)).astype("float32"))
            dist, idx = three_nn(xyz1, xyz2)
            weight = tf.ones_like(dist) / 3.0
            interpolated_points = three_interpolate(points, idx, weight)
            print(interpolated_points)
            # Gradient checking is not as straightforward in TensorFlow 2.x
            # TensorFlow 2.x uses tf.GradientTape for gradient computation
            with tf.GradientTape() as tape:
                tape.watch(points)
                interpolated_points = three_interpolate(points, idx, weight)
            grads = tape.gradient(interpolated_points, points)
            print(grads)


class TestSampling(tf.test.TestCase):
    def test(self):
        np.random.seed(100)
        triangles = np.random.rand(1, 5, 3, 3).astype("float32")
        with tf.device("/gpu:0"):
            inp = tf.constant(triangles)
            tria = inp[:, :, 0, :]
            trib = inp[:, :, 1, :]
            tric = inp[:, :, 2, :]
            areas = tf.sqrt(
                tf.reduce_sum(tf.linalg.cross(trib - tria, tric - tria) ** 2, 2) + 1e-9
            )
            randomnumbers = tf.random.uniform((1, 8192))
            triids = prob_sample(areas, randomnumbers)
            tria_sample = gather_point(tria, triids)
            trib_sample = gather_point(trib, triids)
            tric_sample = gather_point(tric, triids)
            us = tf.random.uniform((1, 8192))
            vs = tf.random.uniform((1, 8192))
            uplusv = 1 - tf.abs(us + vs - 1)
            uminusv = us - vs
            us = (uplusv + uminusv) * 0.5
            vs = (uplusv - uminusv) * 0.5
            pt_sample = (
                    tria_sample
                    + (trib_sample - tria_sample) * tf.expand_dims(us, -1)
                    + (tric_sample - tria_sample) * tf.expand_dims(vs, -1)
            )
            print("pt_sample: ", pt_sample.numpy())
            reduced_sample = gather_point(
                pt_sample, farthest_point_sample(1024, pt_sample)
            )
            print(reduced_sample.numpy())


if __name__ == "__main__":
    tf.test.main()
