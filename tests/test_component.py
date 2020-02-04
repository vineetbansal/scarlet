import pytest
from scarlet.numeric import np
from scarlet.numeric import assert_array_equal, assert_almost_equal

import scarlet

class TestCubeComponent:

    def test_model(self):
        frame_shape = (10, 20, 30)
        frame = scarlet.Frame(frame_shape)

        shape = (5,4,6)
        cube = np.zeros(shape)
        on_location = (1,2,3)
        cube[on_location] = 1
        cube = scarlet.Parameter(cube)
        origin = (2,3,4)
        bbox = scarlet.Box(shape, origin=origin)

        component = scarlet.CubeComponent(frame, cube, bbox=bbox)
        model = component.get_model()

        # everything zero except at one location?
        test_loc = tuple(int(x) for x in (np.array(on_location) + np.array(origin)))
        mask = np.zeros(model.shape, dtype=np.bool)
        mask[test_loc] = True
        assert_array_equal(model[~mask], 0)
        assert model[test_loc] == 1


class TestFactorizedComponent:

    def test_model(self):
        frame_shape = (10, 20, 30)
        frame = scarlet.Frame(frame_shape)

        shape = (5,4,6)
        on_location = (1,2,3)
        sed = np.zeros(shape[0])
        sed[on_location[0]] = 1
        morph = np.zeros(shape[1:])
        morph[on_location[1:]] = 1

        sed = scarlet.Parameter(sed)
        morph = scarlet.Parameter(morph)
        origin = (2,3,4)
        bbox = scarlet.Box(shape, origin=origin)

        component = scarlet.FactorizedComponent(frame, sed, morph, bbox=bbox)
        model = component.get_model()

        # everything zero except at one location?
        test_loc = tuple(int(x) for x in (np.array(on_location) + np.array(origin)))
        mask = np.zeros(model.shape, dtype=np.bool)
        mask[test_loc] = True
        assert_array_equal(model[~mask], 0)
        assert model[test_loc] == 1

        # now with shift
        shift_loc = (0,1,0)
        shift = scarlet.Parameter(np.array(shift_loc[1:]))
        component = scarlet.FactorizedComponent(frame, sed, morph, shift=shift, bbox=bbox)
        model = component.get_model()

        # everything zero except at one location?
        test_loc = tuple(int(x) for x in (np.array(on_location) + np.array(origin) + np.array(shift_loc)))
        mask = np.zeros(model.shape, dtype=np.bool)
        mask[test_loc] = True
        assert_almost_equal(model[~mask], 0)
        assert_almost_equal(model[test_loc],1)

class TestFunctionComponent:

    def test_model(self):
        frame_shape = (10, 20, 30)
        frame = scarlet.Frame(frame_shape)

        shape = (5,4,6)
        on_location = (1,2,3)
        sed = np.zeros(shape[0])
        sed[on_location[0]] = 1

        fparams = np.array(on_location[1:])
        def f(*params):
            morph = np.zeros(shape[1:])
            morph[tuple(int(x) for x in params)] = 1
            return morph

        sed = scarlet.Parameter(sed)
        fparams = scarlet.Parameter(fparams)
        origin = (2,3,4)
        bbox = scarlet.Box(shape, origin=origin)

        component = scarlet.FunctionComponent(frame, sed, fparams, f, bbox=bbox)
        model = component.get_model()

        # everything zero except at one location?
        test_loc = tuple(int(x) for x in (np.array(on_location) + np.array(origin)))
        mask = np.zeros(model.shape, dtype=np.bool)
        mask[test_loc] = True
        assert_array_equal(model[~mask], 0)
        assert model[test_loc] == 1


class TestComponentTree:

    def test_model(self):
        frame_shape = (10, 20, 30)
        frame = scarlet.Frame(frame_shape)

        shape = (5,4,6)
        on_location = (1,2,3)
        cube = np.zeros(shape)
        cube[on_location] = 1
        cube = scarlet.Parameter(cube)
        origin1 = (2,3,4)
        bbox1 = scarlet.Box(shape, origin=origin1)
        component1 = scarlet.CubeComponent(frame, cube, bbox=bbox1)

        sed = np.zeros(shape[0])
        sed[on_location[0]] = 1
        morph = np.zeros(shape[1:])
        morph[on_location[1:]] = 1

        sed = scarlet.Parameter(sed)
        morph = scarlet.Parameter(morph)

        origin2 = (5,6,7)
        bbox2 = scarlet.Box(shape, origin=origin2)
        component2 = scarlet.FactorizedComponent(frame, sed, morph, bbox=bbox2)

        tree = scarlet.ComponentTree([component1, component2])
        model = tree.get_model()

        # everything zero except at one location?
        test_locs = [
            tuple(int(x) for x in (np.array(on_location) + np.array(origin1))),
            tuple(int(x) for x in (np.array(on_location) + np.array(origin2)))
        ]
        mask = np.zeros(model.shape, dtype=np.bool)
        for test_loc in test_locs:
            mask[test_loc] = True
        assert_array_equal(model[~mask], 0)
        assert_array_equal(model[mask], 1)
