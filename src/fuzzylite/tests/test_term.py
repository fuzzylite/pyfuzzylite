import unittest

from fuzzylite.term import *


class TermAssert(object):
    def __init__(self, test: unittest.TestCase, actual: Term):
        self.test = test
        self.actual = actual

    def has_name(self, name, height=1.0):
        self.test.assertEqual(self.actual.name, name)
        self.test.assertEqual(self.actual.height, height)
        return self

    def takes_parameters(self, parameters: int):
        with self.test.assertRaisesRegex(ValueError,
                                         r"not enough values to unpack \(expected %s, got 0\)" % parameters):
            self.actual.__class__().configure("")
        return self

    def is_monotonic(self, monotonic=True):
        self.test.assertEqual(self.actual.is_monotonic(), monotonic)
        return self

    def is_not_monotonic(self):
        self.test.assertEqual(self.actual.is_monotonic(), False)
        return self

    def exports_to(self, fll: str):
        self.test.assertEqual(str(self.actual), fll)
        return self

    def configured_as(self, parameters: str):
        self.actual.configure(parameters)
        return self

    def has_membership(self, x, mf):
        if isnan(mf):
            self.test.assertEqual(isnan(self.actual.membership(x)), True)
        else:
            self.test.assertEqual(self.actual.membership(x), mf)
        return self

    def has_memberships(self, x_mf: dict):
        for x in x_mf.keys():
            self.has_membership(x, x_mf[x])
        return self


class TestTerm(unittest.TestCase):
    def test_term(self):
        self.assertEqual(Term().name, "")
        self.assertEqual(Term("X").name, "X")
        self.assertEqual(Term("X").height, 1.0)
        self.assertEqual(Term("X", .5).height, .5)

        self.assertEqual(str(Term("xxx", 0.5)), "term: xxx Term 0.500")
        self.assertEqual(Term().is_monotonic(), False)

    def test_bell(self):
        TermAssert(self, Bell("bell")) \
            .exports_to("term: bell Bell nan nan nan") \
            .takes_parameters(3) \
            .is_not_monotonic() \
            .configured_as("0 0.25 3.0") \
            .exports_to("term: bell Bell 0.000 0.250 3.000") \
            .has_memberships({-0.5: 0.015384615384615385,
                              -0.4: 0.05625177755617076,
                              -0.25: 0.5,
                              -0.1: 0.9959207087768499,
                              0.0: 1.0,
                              0.1: 0.9959207087768499,
                              0.25: 0.5,
                              0.4: 0.05625177755617076,
                              0.5: 0.015384615384615385,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0}) \
            .configured_as("0 0.25 3.0 0.5") \
            .exports_to("term: bell Bell 0.000 0.250 3.000 0.500") \
            .has_memberships({-0.5: 0.5 * 0.015384615384615385,
                              -0.4: 0.5 * 0.05625177755617076,
                              -0.25: 0.5 * 0.5,
                              -0.1: 0.5 * 0.9959207087768499,
                              0.0: 0.5 * 1.0,
                              0.1: 0.5 * 0.9959207087768499,
                              0.25: 0.5 * 0.5,
                              0.4: 0.5 * 0.05625177755617076,
                              0.5: 0.5 * 0.015384615384615385,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0})

    def test_binary(self):
        TermAssert(self, Binary("binary")) \
            .exports_to("term: binary Binary nan nan") \
            .takes_parameters(2) \
            .is_not_monotonic() \
            .configured_as("0.5 inf") \
            .exports_to("term: binary Binary 0.500 inf") \
            .has_memberships({0.0: 0.0,
                              0.1: 0.0,
                              0.25: 0.0,
                              0.49: 0.0,
                              0.5: 1.0,
                              0.51: 1.0,
                              0.75: 1.0,
                              0.9: 1.0,
                              1.0: 1.0,
                              nan: nan,
                              inf: 1.0,
                              -inf: 0.0}) \
            .configured_as("0.5 -inf 0.5") \
            .exports_to("term: binary Binary 0.500 -inf 0.500") \
            .has_memberships({0.0: 0.5,
                              0.1: 0.5,
                              0.25: 0.5,
                              0.49: 0.5,
                              0.5: 0.5,
                              0.51: 0.0,
                              0.75: 0.0,
                              0.9: 0.0,
                              1.0: 0.0,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.5})

    def test_concave(self):
        TermAssert(self, Concave("concave")) \
            .exports_to("term: concave Concave nan nan") \
            .takes_parameters(2) \
            .is_monotonic() \
            .configured_as("0.500 0.750") \
            .exports_to("term: concave Concave 0.500 0.750") \
            .has_memberships({0.0: 0.250,
                              0.1: 0.2777777777777778,
                              0.25: 0.3333333333333333,
                              0.4: 0.4166666666666667,
                              0.5: 0.5,
                              0.6: 0.625,
                              0.75: 1.0,
                              0.9: 1.0,
                              1.0: 1.0,
                              nan: nan,
                              inf: 1.0,
                              -inf: 0.0}) \
            .configured_as("0.500 0.250 0.5") \
            .exports_to("term: concave Concave 0.500 0.250 0.500") \
            .has_memberships({0.0: 0.5,
                              0.1: 0.5,
                              0.25: 0.5,
                              0.4: 0.5 * 0.625,
                              0.5: 0.5 * 0.5,
                              0.6: 0.5 * 0.4166666666666667,
                              0.75: 0.5 * 0.3333333333333333,
                              0.9: 0.5 * 0.2777777777777778,
                              1.0: 0.5 * 0.250,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.5})

    def test_constant(self):
        TermAssert(self, Constant("constant")) \
            .exports_to("term: constant Constant nan") \
            .takes_parameters(1) \
            .is_not_monotonic() \
            .configured_as("0.5") \
            .exports_to("term: constant Constant 0.500") \
            .has_memberships({0.0: 0.5,
                              0.1: 0.5,
                              0.25: 0.5,
                              0.4: 0.5,
                              0.5: 0.5,
                              0.6: 0.5,
                              0.75: 0.5,
                              0.9: 0.5,
                              1.0: 0.5,
                              nan: 0.5,
                              inf: 0.5,
                              -inf: 0.5}) \
            .configured_as("0.500 0.5") \
            .exports_to("term: constant Constant 0.500") \
            .has_memberships({0.0: 0.5,
                              0.1: 0.5,
                              0.25: 0.5,
                              0.4: 0.5,
                              0.5: 0.5,
                              0.6: 0.5,
                              0.75: 0.5,
                              0.9: 0.5,
                              1.0: 0.5,
                              nan: 0.5,
                              inf: 0.5,
                              -inf: 0.5})

    def test_cosine(self):
        TermAssert(self, Cosine("cosine")) \
            .exports_to("term: cosine Cosine nan nan") \
            .takes_parameters(2) \
            .is_not_monotonic() \
            .configured_as("0.5 1") \
            .exports_to("term: cosine Cosine 0.500 1.000") \
            .has_memberships({0.0: 0.0,
                              0.1: 0.09549150281252633,
                              0.25: 0.5,
                              0.49: 0.9990133642141358,
                              0.5: 1.0,
                              0.51: 0.9990133642141358,
                              0.75: 0.5,
                              0.9: 0.09549150281252633,
                              1.0: 0.0,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0}) \
            .configured_as("0.5 1.0 0.5") \
            .exports_to("term: cosine Cosine 0.500 1.000 0.500") \
            .has_memberships({0.0: 0.5 * 0.0,
                              0.1: 0.5 * 0.09549150281252633,
                              0.25: 0.5 * 0.5,
                              0.49: 0.5 * 0.9990133642141358,
                              0.5: 0.5 * 1.0,
                              0.51: 0.5 * 0.9990133642141358,
                              0.75: 0.5 * 0.5,
                              0.9: 0.5 * 0.09549150281252633,
                              1.0: 0.5 * 0.0,
                              nan: 0.5 * nan,
                              inf: 0.5 * 0.0,
                              -inf: 0.5 * 0.0})

    def test_discrete(self):
        pass

    def test_function(self):
        pass

    def test_gaussian(self):
        TermAssert(self, Gaussian("gaussian")) \
            .exports_to("term: gaussian Gaussian nan nan") \
            .takes_parameters(2) \
            .is_not_monotonic() \
            .configured_as("0.5 0.25") \
            .exports_to("term: gaussian Gaussian 0.500 0.250") \
            .has_memberships({0.0: 0.1353352832366127,
                              0.1: 0.2780373004531941,
                              0.25: 0.6065306597126334,
                              0.49: 0.9992003199146837,
                              0.5: 1.0,
                              0.51: 0.9992003199146837,
                              0.75: 0.6065306597126334,
                              0.9: 0.2780373004531941,
                              1.0: 0.1353352832366127,
                              nan: nan,
                              inf: 0.0,
                              -inf: 0.0}) \
            .configured_as("0.5 0.25 0.5") \
            .exports_to("term: gaussian Gaussian 0.500 0.250 0.500") \
            .has_memberships({0.0: 0.5 * 0.1353352832366127,
                              0.1: 0.5 * 0.2780373004531941,
                              0.25: 0.5 * 0.6065306597126334,
                              0.49: 0.5 * 0.9992003199146837,
                              0.5: 0.5 * 1.0,
                              0.51: 0.5 * 0.9992003199146837,
                              0.75: 0.5 * 0.6065306597126334,
                              0.9: 0.5 * 0.2780373004531941,
                              1.0: 0.5 * 0.1353352832366127,
                              nan: 0.5 * nan,
                              inf: 0.5 * 0.0,
                              -inf: 0.5 * 0.0})

        self.assertEqual(str(Gaussian()), "term: unnamed Gaussian nan nan")
        self.assertEqual(str(Gaussian("x", 0.5, 1)), "term: x Gaussian 0.500 1.000")
        self.assertEqual(str(Gaussian("x", 0.5, 1, 0.5)), "term: x Gaussian 0.500 1.000 0.500")
        self.assertEqual(Gaussian().is_monotonic(), False)

        with self.assertRaisesRegex(ValueError, r"not enough values to unpack \(expected 2, got 0\)"):
            Gaussian().configure("")

        gaussian = Gaussian("gaussian")
        gaussian.configure("0.5 0.25")
        self.assertEqual(str(gaussian), "term: gaussian Gaussian 0.500 0.250")
        self.assertEqual(gaussian.membership(0.5), 1.0)
        self.assertEqual(gaussian.membership(0.75), 0.6065306597126334)
        self.assertEqual(gaussian.membership(0.25), 0.6065306597126334)
        gaussian.configure("0.5 0.25 0.5")
        self.assertEqual(gaussian.membership(0.5), 0.5)
        self.assertEqual(gaussian.membership(0.75), 0.3032653298563167)
        self.assertEqual(gaussian.membership(0.25), 0.3032653298563167)

    def test_gaussian_product(self):
        self.assertEqual(str(GaussianProduct()), "term: unnamed GaussianProduct nan nan nan nan")
        self.assertEqual(str(GaussianProduct("x", 0.3, 0.1, 0.6, 0.2)),
                         "term: x GaussianProduct 0.300 0.100 0.600 0.200")
        self.assertEqual(str(GaussianProduct("x", 0.3, 0.1, 0.6, 0.2, 0.5)),
                         "term: x GaussianProduct 0.300 0.100 0.600 0.200 0.500")
        self.assertEqual(GaussianProduct().is_monotonic(), False)

        with self.assertRaisesRegex(ValueError, r"not enough values to unpack \(expected 4, got 0\)"):
            GaussianProduct().configure("")

        gaussian_product = GaussianProduct("gaussianProduct")
        gaussian_product.configure("0.30 0.10 0.60 0.20")
        self.assertEqual(str(gaussian_product), "term: gaussianProduct GaussianProduct 0.300 0.100 0.600 0.200")
        self.assertEqual(gaussian_product.membership(0.5), 1.0)
        self.assertEqual(gaussian_product.membership(0.75), 0.7548396019890073)
        self.assertEqual(gaussian_product.membership(0.25), 0.8824969025845955)
        gaussian_product.configure("0.30 0.10 0.60 0.20 0.5")
        self.assertEqual(gaussian_product.membership(0.5), 0.5)
        self.assertEqual(gaussian_product.membership(0.75), 0.3774198009945037)
        self.assertEqual(gaussian_product.membership(0.25), 0.4412484512922977)

    def test_pi_shape(self):
        self.assertEqual(str(PiShape()), "term: unnamed PiShape nan nan nan nan")
        self.assertEqual(str(PiShape("x", 0.000, 0.333, 0.666, 1.000)),
                         "term: x PiShape 0.000 0.333 0.666 1.000")
        self.assertEqual(str(PiShape("x", 0.000, 0.333, 0.666, 1.000, 0.5)),
                         "term: x PiShape 0.000 0.333 0.666 1.000 0.500")
        self.assertEqual(PiShape().is_monotonic(), False)

        with self.assertRaisesRegex(ValueError, r"not enough values to unpack \(expected 4, got 0\)"):
            PiShape().configure("")

        pi_shape = PiShape("pi_shape")
        pi_shape.configure("0.000 0.333 0.666 1.000")
        self.assertEqual(str(pi_shape), "term: pi_shape PiShape 0.000 0.333 0.666 1.000")
        self.assertEqual(pi_shape.membership(0.5), 1.0)
        self.assertEqual(pi_shape.membership(0.75), 0.8734985119581198)
        self.assertEqual(pi_shape.membership(0.25), 0.8757496234973712)
        pi_shape.configure("0.000 0.333 0.666 1.000 0.5")
        self.assertEqual(pi_shape.membership(0.5), 0.5)
        self.assertEqual(pi_shape.membership(0.75), 0.4367492559790599)
        self.assertEqual(pi_shape.membership(0.25), 0.4378748117486856)

    def test_ramp(self):
        self.assertEqual(str(Ramp()), "term: unnamed Ramp nan nan")
        self.assertEqual(str(Ramp("x", 0.25, 0.750)), "term: x Ramp 0.250 0.750")
        self.assertEqual(str(Ramp("x", 0.25, 0.750, 0.5)), "term: x Ramp 0.250 0.750 0.500")
        self.assertEqual(Ramp().is_monotonic(), False)

        with self.assertRaisesRegex(ValueError, r"not enough values to unpack \(expected 2, got 0\)"):
            Ramp().configure("")

        ramp = Ramp("ramp")
        ramp.configure("0.250 0.750")
        self.assertEqual(str(ramp), "term: ramp Ramp 0.250 0.750")
        self.assertEqual(ramp.membership(0.5), 0.5)
        self.assertEqual(ramp.membership(0.75), 1.0)
        self.assertEqual(ramp.membership(0.6), 0.7)
        self.assertEqual(ramp.membership(0.25), 0.0)
        ramp.configure("0.250 0.750 0.5")
        self.assertEqual(ramp.membership(0.5), 0.25)
        self.assertEqual(ramp.membership(0.75), 0.5)
        self.assertEqual(ramp.membership(0.6), 0.35)
        self.assertEqual(ramp.membership(0.25), 0.0)

    def test_rectangle(self):
        self.assertEqual(str(Rectangle()), "term: unnamed Rectangle nan nan")
        self.assertEqual(str(Rectangle("x", 0.25, 0.750)), "term: x Rectangle 0.250 0.750")
        self.assertEqual(str(Rectangle("x", 0.25, 0.750, 0.5)), "term: x Rectangle 0.250 0.750 0.500")
        self.assertEqual(Rectangle().is_monotonic(), False)

        with self.assertRaisesRegex(ValueError, r"not enough values to unpack \(expected 2, got 0\)"):
            Rectangle().configure("")

        rectangle = Rectangle("rectangle")
        rectangle.configure("0.250 0.750")
        self.assertEqual(str(rectangle), "term: rectangle Rectangle 0.250 0.750")
        self.assertEqual(rectangle.membership(0.5), 1.0)
        self.assertEqual(rectangle.membership(0.75), 1.0)
        self.assertEqual(rectangle.membership(0.25), 1.0)
        self.assertEqual(rectangle.membership(0.76), 0.0)
        self.assertEqual(rectangle.membership(0.24), 0.0)
        rectangle.configure("0.250 0.750 0.5")
        self.assertEqual(rectangle.membership(0.5), 0.5)
        self.assertEqual(rectangle.membership(0.75), 0.5)
        self.assertEqual(rectangle.membership(0.25), 0.5)
        self.assertEqual(rectangle.membership(0.76), 0.0)
        self.assertEqual(rectangle.membership(0.24), 0.0)

    def test_s_shape(self):
        self.assertEqual(str(SShape()), "term: unnamed SShape nan nan")
        self.assertEqual(str(SShape("x", 0.25, 0.750)), "term: x SShape 0.250 0.750")
        self.assertEqual(str(SShape("x", 0.25, 0.750, 0.5)), "term: x SShape 0.250 0.750 0.500")
        self.assertEqual(SShape().is_monotonic(), False)

        with self.assertRaisesRegex(ValueError, r"not enough values to unpack \(expected 2, got 0\)"):
            SShape().configure("")

        s_shape = SShape("s_shape")
        s_shape.configure("0.250 0.750")
        self.assertEqual(str(s_shape), "term: s_shape SShape 0.250 0.750")
        self.assertEqual(s_shape.membership(0.5), 0.5)
        self.assertEqual(s_shape.membership(0.6), 0.82)
        self.assertEqual(s_shape.membership(0.75), 1.0)
        self.assertEqual(s_shape.membership(0.25), 0.0)
        s_shape.configure("0.250 0.750 0.5")
        self.assertEqual(s_shape.membership(0.5), 0.25)
        self.assertEqual(s_shape.membership(0.6), 0.41)
        self.assertEqual(s_shape.membership(0.75), 0.5)
        self.assertEqual(s_shape.membership(0.25), 0.0)

    def test_sigmoid(self):
        self.assertEqual(str(Sigmoid()), "term: unnamed Sigmoid nan nan")
        self.assertEqual(str(Sigmoid("x", 0.5, 40.000)), "term: x Sigmoid 0.500 40.000")
        self.assertEqual(str(Sigmoid("x", 0.5, 40.000, 0.5)), "term: x Sigmoid 0.500 40.000 0.500")
        self.assertEqual(Sigmoid().is_monotonic(), False)

        with self.assertRaisesRegex(ValueError, r"not enough values to unpack \(expected 2, got 0\)"):
            Sigmoid().configure("")

        sigmoid = Sigmoid("sigmoid")
        sigmoid.configure("0.50 40.000")
        self.assertEqual(str(sigmoid), "term: sigmoid Sigmoid 0.500 40.000")
        self.assertEqual(sigmoid.membership(0.25), 4.5397868702434395e-05)
        self.assertEqual(sigmoid.membership(0.4), 0.017986209962091573)
        self.assertEqual(sigmoid.membership(0.5), 0.5)
        self.assertEqual(sigmoid.membership(0.6), 0.9820137900379085)
        self.assertEqual(sigmoid.membership(0.75), 0.9999546021312976)

        sigmoid.configure("0.50 40.000 0.5")
        self.assertEqual(sigmoid.membership(0.25), 2.2698934351217197e-05)
        self.assertEqual(sigmoid.membership(0.4), 0.008993104981045786)
        self.assertEqual(sigmoid.membership(0.5), 0.25)
        self.assertEqual(sigmoid.membership(0.6), 0.4910068950189542)
        self.assertEqual(sigmoid.membership(0.75), 0.4999773010656488)

    def test_sigmoid_difference(self):
        self.assertEqual(str(SigmoidDifference()), "term: unnamed SigmoidDifference nan nan nan nan")
        self.assertEqual(str(SigmoidDifference("x", 0.25, 40, 20, 0.75)),
                         "term: x SigmoidDifference 0.250 40.000 20.000 0.750")
        self.assertEqual(str(SigmoidDifference("x", 0.25, 40, 20, 0.75, 0.5)),
                         "term: x SigmoidDifference 0.250 40.000 20.000 0.750 0.500")
        self.assertEqual(SigmoidDifference().is_monotonic(), False)

        with self.assertRaisesRegex(ValueError, r"not enough values to unpack \(expected 4, got 0\)"):
            SigmoidDifference().configure("")

        sigmoid_difference = SigmoidDifference("sigmoid_difference")
        sigmoid_difference.configure("0.25 40 20 0.75")
        self.assertEqual(str(sigmoid_difference),
                         "term: sigmoid_difference SigmoidDifference 0.250 40.000 20.000 0.750")
        self.assertEqual(sigmoid_difference.membership(0.25), 0.49995460213129755)
        self.assertEqual(sigmoid_difference.membership(0.4), 0.9966163256489647)
        self.assertEqual(sigmoid_difference.membership(0.5), 0.9932617512070128)
        self.assertEqual(sigmoid_difference.membership(0.6), 0.9525732952944055)
        self.assertEqual(sigmoid_difference.membership(0.75), 0.4999999979388463)

        sigmoid_difference.configure("0.25 40 20 0.75 0.5")
        self.assertEqual(sigmoid_difference.membership(0.25), 0.24997730106564878)
        self.assertEqual(sigmoid_difference.membership(0.4), 0.49830816282448237)
        self.assertEqual(sigmoid_difference.membership(0.5), 0.4966308756035064)
        self.assertEqual(sigmoid_difference.membership(0.6), 0.47628664764720274)
        self.assertEqual(sigmoid_difference.membership(0.75), 0.24999999896942315)

    def test_sigmoid_product(self):
        self.assertEqual(str(SigmoidProduct()), "term: unnamed SigmoidProduct nan nan nan nan")
        self.assertEqual(str(SigmoidProduct("x", 0.25, 40, -20, 0.75)),
                         "term: x SigmoidProduct 0.250 40.000 -20.000 0.750")
        self.assertEqual(str(SigmoidProduct("x", 0.25, 40, -20, 0.75, 0.5)),
                         "term: x SigmoidProduct 0.250 40.000 -20.000 0.750 0.500")
        self.assertEqual(SigmoidProduct().is_monotonic(), False)

        with self.assertRaisesRegex(ValueError, r"not enough values to unpack \(expected 4, got 0\)"):
            SigmoidProduct().configure("")

        sigmoid_product = SigmoidProduct("sigmoid_product")
        sigmoid_product.configure("0.25 40 -20 0.75")
        self.assertEqual(str(sigmoid_product),
                         "term: sigmoid_product SigmoidProduct 0.250 40.000 -20.000 0.750")
        self.assertEqual(sigmoid_product.membership(0.25), 0.4999773010656488)
        self.assertEqual(sigmoid_product.membership(0.4), 0.9966185783352449)
        self.assertEqual(sigmoid_product.membership(0.5), 0.9932620550481802)
        self.assertEqual(sigmoid_product.membership(0.6), 0.9525733347303482)
        self.assertEqual(sigmoid_product.membership(0.75), 0.49999999896942315)

        sigmoid_product.configure("0.25 40 -20 0.75 0.5")
        self.assertEqual(sigmoid_product.membership(0.25), 0.2499886505328244)
        self.assertEqual(sigmoid_product.membership(0.4), 0.49830928916762246)
        self.assertEqual(sigmoid_product.membership(0.5), 0.4966310275240901)
        self.assertEqual(sigmoid_product.membership(0.6), 0.4762866673651741)
        self.assertEqual(sigmoid_product.membership(0.75), 0.24999999948471158)

    def test_spike(self):
        self.assertEqual(str(Spike()), "term: unnamed Spike nan nan")
        self.assertEqual(str(Spike("x", 0.5, 1.0)), "term: x Spike 0.500 1.000")
        self.assertEqual(str(Spike("x", 0.5, 1.0, 0.5)), "term: x Spike 0.500 1.000 0.500")
        self.assertEqual(Spike().is_monotonic(), False)

        with self.assertRaisesRegex(ValueError, r"not enough values to unpack \(expected 2, got 0\)"):
            Spike().configure("")

        spike = Spike("spike")
        spike.configure("0.500 1.000")
        self.assertEqual(str(spike), "term: spike Spike 0.500 1.000")
        self.assertEqual(spike.membership(0.25), 0.0820849986238988)
        self.assertEqual(spike.membership(0.4), 0.3678794411714424)
        self.assertEqual(spike.membership(0.5), 1)
        self.assertEqual(spike.membership(0.6), 0.3678794411714424)
        self.assertEqual(spike.membership(0.75), 0.0820849986238988)
        spike.configure("0.500 1.000 0.5")
        self.assertEqual(spike.membership(0.25), 0.0410424993119494)
        self.assertEqual(spike.membership(0.4), 0.1839397205857212)
        self.assertEqual(spike.membership(0.5), 0.5)
        self.assertEqual(spike.membership(0.6), 0.1839397205857212)
        self.assertEqual(spike.membership(0.75), 0.0410424993119494)

    def test_trapezoid(self):
        self.assertEqual(str(Trapezoid()), "term: unnamed Trapezoid nan nan nan nan")
        self.assertEqual(str(Trapezoid("x", 0.1, 0.3, 0.7, 0.9)),
                         "term: x Trapezoid 0.100 0.300 0.700 0.900")
        self.assertEqual(str(Trapezoid("x", 0.1, 0.3, 0.7, 0.9, 0.5)),
                         "term: x Trapezoid 0.100 0.300 0.700 0.900 0.500")
        self.assertEqual(Trapezoid().is_monotonic(), False)

        with self.assertRaisesRegex(ValueError, r"not enough values to unpack \(expected 4, got 0\)"):
            Trapezoid().configure("")

        trapezoid = Trapezoid("trapezoid")
        trapezoid.configure("0.100 0.300 0.700 0.900")
        self.assertEqual(str(trapezoid),
                         "term: trapezoid Trapezoid 0.100 0.300 0.700 0.900")
        self.assertEqual(trapezoid.membership(0.1), 0.0)
        self.assertEqual(trapezoid.membership(0.25), 0.75)
        self.assertEqual(trapezoid.membership(0.4), 1.0)
        self.assertEqual(trapezoid.membership(0.5), 1.0)
        self.assertEqual(trapezoid.membership(0.6), 1.0)
        self.assertEqual(trapezoid.membership(0.75), 0.7499999999999999)
        self.assertEqual(trapezoid.membership(0.9), 0.0)

        trapezoid.configure("0.100 0.300 0.700 0.900 0.500")
        self.assertEqual(trapezoid.membership(0.1), 0.0)
        self.assertEqual(trapezoid.membership(0.25), 0.375)
        self.assertEqual(trapezoid.membership(0.4), 0.5)
        self.assertEqual(trapezoid.membership(0.5), 0.5)
        self.assertEqual(trapezoid.membership(0.6), 0.5)
        self.assertEqual(trapezoid.membership(0.75), 0.37499999999999994)
        self.assertEqual(trapezoid.membership(0.9), 0.0)

    def test_triangle(self):
        self.assertEqual(str(Triangle()), "term: unnamed Triangle nan nan nan")
        self.assertEqual(str(Triangle("x", 0.25, 0.5, 0.75)),
                         "term: x Triangle 0.250 0.500 0.750")
        self.assertEqual(str(Triangle("x", 0.25, 0.5, 0.75, 0.5)),
                         "term: x Triangle 0.250 0.500 0.750 0.500")
        self.assertEqual(Triangle().is_monotonic(), False)

        with self.assertRaisesRegex(ValueError, r"not enough values to unpack \(expected 3, got 0\)"):
            Triangle().configure("")

        triangle = Triangle("triangle")
        triangle.configure("0.250 0.500 0.750")
        self.assertEqual(str(triangle), "term: triangle Triangle 0.250 0.500 0.750")
        self.assertEqual(triangle.membership(0.0), 0.0)
        self.assertEqual(triangle.membership(0.1), 0.0)
        self.assertEqual(triangle.membership(0.25), 0.0)
        self.assertEqual(triangle.membership(0.4), 0.6000000000000001)
        self.assertEqual(triangle.membership(0.5), 1.0)
        self.assertEqual(triangle.membership(0.6), 0.6000000000000001)
        self.assertEqual(triangle.membership(0.75), 0.0)
        self.assertEqual(triangle.membership(0.9), 0.0)
        self.assertEqual(triangle.membership(1.0), 0.0)

        triangle.configure("0.250 0.500 0.750 0.500")
        self.assertEqual(triangle.membership(0.0), 0.0)
        self.assertEqual(triangle.membership(0.1), 0.0)
        self.assertEqual(triangle.membership(0.25), 0.0)
        self.assertEqual(triangle.membership(0.4), 0.30000000000000004)
        self.assertEqual(triangle.membership(0.5), 0.5)
        self.assertEqual(triangle.membership(0.6), 0.30000000000000004)
        self.assertEqual(triangle.membership(0.75), 0.0)
        self.assertEqual(triangle.membership(0.9), 0.0)
        self.assertEqual(triangle.membership(1.0), 0.0)

    def test_z_shape(self):
        self.assertEqual(str(ZShape()), "term: unnamed ZShape nan nan")
        self.assertEqual(str(ZShape("x", 0.25, 0.750)), "term: x ZShape 0.250 0.750")
        self.assertEqual(str(ZShape("x", 0.25, 0.750, 0.5)), "term: x ZShape 0.250 0.750 0.500")
        self.assertEqual(ZShape().is_monotonic(), False)

        with self.assertRaisesRegex(ValueError, r"not enough values to unpack \(expected 2, got 0\)"):
            ZShape().configure("")

        z_shape = ZShape("z_shape")
        z_shape.configure("0.250 0.750")
        self.assertEqual(str(z_shape), "term: z_shape ZShape 0.250 0.750")
        self.assertEqual(z_shape.membership(0.5), 0.5)
        self.assertEqual(z_shape.membership(0.6), 0.18000000000000005)
        self.assertEqual(z_shape.membership(0.75), 0.0)
        self.assertEqual(z_shape.membership(0.25), 1.0)
        z_shape.configure("0.250 0.750 0.5")
        self.assertEqual(z_shape.membership(0.5), 0.25)
        self.assertEqual(z_shape.membership(0.6), 0.09000000000000002)
        self.assertEqual(z_shape.membership(0.75), 0.0)
        self.assertEqual(z_shape.membership(0.25), 0.5)


if __name__ == '__main__':
    unittest.main()
