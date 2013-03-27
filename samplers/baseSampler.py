import numpy
from theano import tensor
from xylearn.utils import toSharedX

if 0:
    print 'WARNING: using SLOW rng'
    RandomStreams = tensor.shared_randomstreams.RandomStreams
else:
    import theano.sandbox.rng_mrg

    RandomStreams = theano.sandbox.rng_mrg.MRG_RandomStreams


class baseSampler(object):
    """
    A sampler is responsible for implementing a sampling strategy on top of
    an RBM, which may include retaining state e.g. the negative particles for
    Persistent Contrastive Divergence.
    """

    def __init__(self, rbm, particles, rng):
        """
        Construct a Sampler.

        Parameters
        ----------
        rbm : object
            An instance of `RBM` or a derived class, or one implementing
            the `gibbs_step_for_v` interface.
        particles : ndarray
            An initial state for the set of persistent Narkov chain particles
            that will be updated at every step of learning.
        rng : RandomState object
            NumPy random number generator object used to initialize a
            RandomStreams object used in training.
        """
        self.__dict__.update(rbm=rbm)
        if not hasattr(rng, 'randn'):
            rng = numpy.random.RandomState(rng)
        seed = int(rng.randint(2 ** 30))
        self.s_rng = RandomStreams(seed)
        self.particles = toSharedX(particles, name='particles')

    def updates(self):
        """
        Get the dictionary of updates for the sampler's persistent state
        at each step.

        Returns
        -------
        updates : dict
            Dictionary with shared variable instances as keys and symbolic
            expressions indicating how they should be updated as values.

        Notes
        -----
        In the `Sampler` base class, this is simply a stub.
        """
        raise NotImplementedError()