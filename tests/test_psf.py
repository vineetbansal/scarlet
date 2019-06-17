import numpy as np
import scarlet
from numpy.testing import assert_array_equal, assert_almost_equal


class TestPsf(object):
    def test_moffat(self):
        shape = 7, 7
        X = np.arange(shape[1])
        Y = np.arange(shape[0])
        X, Y = np.meshgrid(X, Y)
        coords = np.stack([Y, X])
        y0 = 3
        x0 = 5
        amplitude = 4.6
        alpha = 1.123
        beta = 1.491
        result = scarlet.psf.moffat(coords, y0, x0, amplitude, alpha, beta)

        truth = [[0.03206059739715075, 0.049750104330030576, 0.07898230258561093, 0.12363652428535422,
                  0.17582539510961576, 0.20197531771865862, 0.17582539510961576],
                 [0.040270826284902264, 0.06816462627549429, 0.12363652428535422, 0.23533923632514342,
                  0.42187244606329377, 0.5468416052781302, 0.42187244606329377],
                 [0.047053767539206766, 0.08551702357967202, 0.17582539510961576, 0.42187244606329377,
                  1.1157287111346086, 1.9261547802252645, 1.1157287111346086],
                 [0.049750104330030576, 0.09300795388382317, 0.20197531771865862, 0.5468416052781302,
                  1.9261547802252645, 4.6, 1.9261547802252645],
                 [0.047053767539206766, 0.08551702357967202, 0.17582539510961576, 0.42187244606329377,
                  1.1157287111346086, 1.9261547802252645, 1.1157287111346086],
                 [0.040270826284902264, 0.06816462627549429, 0.12363652428535422, 0.23533923632514342,
                  0.42187244606329377, 0.5468416052781302, 0.42187244606329377],
                 [0.03206059739715075, 0.049750104330030576, 0.07898230258561093, 0.12363652428535422,
                  0.17582539510961576, 0.20197531771865862, 0.17582539510961576]]
        assert_almost_equal(result, truth)

    def test_gaussian(self):
        shape = 7, 7
        X = np.arange(shape[1])
        Y = np.arange(shape[0])
        X, Y = np.meshgrid(X, Y)
        coords = np.stack([Y, X])
        y0 = 3
        x0 = 5
        amplitude = 4.6
        sigma = 1.872
        result = scarlet.psf.gaussian(coords, y0, x0, amplitude, sigma)

        truth = [[0.035972150170478605, 0.1299111667397572, 0.3526936664142025, 0.7198134054492321,
                  1.1043666731044601, 1.2737310805289048, 1.1043666731044601],
                 [0.07341565324605076, 0.26513603231822036, 0.7198134054492321, 1.469069019390693,
                  2.253904766462087, 2.5995610185562463, 2.253904766462087],
                 [0.11263724753574361, 0.4067823629217491, 1.1043666731044601, 2.253904766462087,
                  3.458031330881594, 3.988351053011173, 3.458031330881594],
                 [0.1299111667397572, 0.46916603994207223, 1.2737310805289048, 2.5995610185562463,
                  3.988351053011173, 4.6, 3.988351053011173],
                 [0.11263724753574361, 0.4067823629217491, 1.1043666731044601, 2.253904766462087,
                  3.458031330881594, 3.988351053011173, 3.458031330881594],
                 [0.07341565324605076, 0.26513603231822036, 0.7198134054492321, 1.469069019390693,
                  2.253904766462087, 2.5995610185562463, 2.253904766462087],
                 [0.035972150170478605, 0.1299111667397572, 0.3526936664142025, 0.7198134054492321,
                  1.1043666731044601, 1.2737310805289048, 1.1043666731044601]]

        assert_almost_equal(result, truth)

    def test_double_gaussian(self):
        shape = 7, 7
        X = np.arange(shape[1])
        Y = np.arange(shape[0])
        X, Y = np.meshgrid(X, Y)
        coords = np.stack([Y, X])
        y0 = 3
        x0 = 5
        A1 = 4.2
        sigma1 = .938
        A2 = 2.042
        sigma2 = 3.67
        result = scarlet.psf.double_gaussian(coords, y0, x0, A1, sigma1, A2, sigma2)

        truth = [[0.5779677726051379, 0.8072428930726728, 1.0469367888293277, 1.2628823397211582,
                  1.4230484212789571, 1.4872678517124784, 1.4230484212789571],
                 [0.6958480260541855, 0.9719301516964072, 1.2628823397211582, 1.5618740725080151,
                  1.9411166635954085, 2.1927762549197896, 1.9411166635954085],
                 [0.7778242508142483, 1.0866424571552145, 1.4230484212789571, 1.9411166635954085,
                  3.24374453148386, 4.34687513808105, 3.24374453148386],
                 [0.8072428930726728, 1.1279342108343222, 1.4872678517124784, 2.1927762549197896,
                  4.34687513808105, 6.242, 4.34687513808105],
                 [0.7778242508142483, 1.0866424571552145, 1.4230484212789571, 1.9411166635954085,
                  3.24374453148386, 4.34687513808105, 3.24374453148386],
                 [0.6958480260541855, 0.9719301516964072, 1.2628823397211582, 1.5618740725080151,
                  1.9411166635954085, 2.1927762549197896, 1.9411166635954085],
                 [0.5779677726051379, 0.8072428930726728, 1.0469367888293277, 1.2628823397211582,
                  1.4230484212789571, 1.4872678517124784, 1.4230484212789571]]

        assert_almost_equal(result, truth)

    def test_fit_target_psf(self):
        shape = 21, 21
        X = np.arange(shape[1])
        Y = np.arange(shape[0])
        X, Y = np.meshgrid(X, Y)
        coords = np.stack([Y, X])
        y0 = 10
        x0 = 10
        amplitude = 4.6
        sigmas = [1.1, 0.91, 2.6, 1.02, 3.5]
        psfs = np.array([scarlet.psf.gaussian(coords, y0, x0, amplitude, sigma) for sigma in sigmas])
        target_psf, all_params, params = scarlet.psf.fit_target_psf(psfs, scarlet.psf.gaussian)

        assert_almost_equal(target_psf.sum(), 1)

        true_all_params = [[10.0, 10.0, 4.6, 1.1],
                           [10.0, 10.0, 4.6, 0.91],
                           [10.0, 10.0, 4.6, 2.6],
                           [10.0, 10.0, 4.6, 1.02],
                           [10.0, 10.0, 4.6, 3.5]]
        assert_almost_equal(all_params, true_all_params)

        true_params = [4.6, 0.546]
        assert_almost_equal(params, true_params)
