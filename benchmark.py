import benchmark

import math

class Benchmark_Sqrt(benchmark.Benchmark):

    each = 100 # allows for differing number of runs

    def setUp(self):
        # Only using setUp in order to subclass later
        # Can also specify tearDown, eachSetUp, and eachTearDown
        self.size = 25000

    def test_pow_operator(self):
        for i in xrange(self.size):
            z = i**.5

    def test_pow_function(self):
        for i in xrange(self.size):
            z = pow(i, .5)

    def test_sqrt_function(self):
        for i in xrange(self.size):
            z = math.sqrt(i)


if __name__ == '__main__':
    benchmark.main(format="markdown", numberFormat="%.4g")
    # could have written benchmark.main(each=50) if the
    # first class shouldn't have been run 100 times.
