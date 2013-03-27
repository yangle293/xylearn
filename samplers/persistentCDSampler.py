from theano import tensor
import numpy
from xylearn.utils import toSharedX
from samplers.baseSampler import baseSampler


class PersistentCDSampler(baseSampler):
    """
        Implements a persistent Markov chain for use with Persistent Contrastive
        Divergence, a.k.a. stochastic maximum likelhiood, as described in [1].
        
        .. [1] T. Tieleman. "Training Restricted Boltzmann Machines using
        approximations to the likelihood gradient". Proceedings of the 25th
        International Conference on Machine Learning, Helsinki, Finland,
        2008. http://www.cs.toronto.edu/~tijmen/pcd/pcd.pdf
        """

    def __init__(self, rbm, particles, rng, steps=1, particles_clip=None):
        """
            Construct a PersistentCDSampler.
            
            Parameters
            ----------
            rbm : object
            An instance of `RBM` or a derived class, or one implementing
            the `gibbs_step_for_v` interface.
            particles : ndarray
            An initial state for the set of persistent Markov chain particles
            that will be updated at every step of learning.
            rng : RandomState object
            NumPy random number generator object used to initialize a
            RandomStreams object used in training.
            steps : int, optional
            Number of Gibbs steps to run the Markov chain for at each
            iteration.
            particles_clip: None or (min, max) pair
            The values of the returned particles will be clipped between
            min and max.
            """
        super(PersistentCDSampler, self).__init__(rbm, particles, rng)
        self.steps = steps
        self.particles_clip = particles_clip

    def updates(self, particles_clip=None):
        """
            Get the dictionary of updates for the sampler's persistent state
            at each step..
            
            Returns
            -------
            updates : dict
            Dictionary with shared variable instances as keys and symbolic
            expressions indicating how they should be updated as values.
            """
        steps = self.steps
        particles = self.particles
        # TODO: do this with scan?
        for i in xrange(steps):
            particles, _locals = self.rbm.gibbs_step_for_v(
                particles,
                self.s_rng
            )
            if self.particles_clip is not None:
                p_min, p_max = self.particles_clip
                # The clipped values should still have the same type
                dtype = particles.dtype
                p_min = tensor.as_tensor_variable(p_min)
                if p_min.dtype != dtype:
                    p_min = tensor.cast(p_min, dtype)
                p_max = tensor.as_tensor_variable(p_max)
                if p_max.dtype != dtype:
                    p_max = tensor.cast(p_max, dtype)
                particles = tensor.clip(particles, p_min, p_max)
        if not hasattr(self.rbm, 'h_sample'):
            self.rbm.h_sample = toSharedX(numpy.zeros((0, 0)), 'h_sample')
        return {
            self.particles: particles,
            # TODO: self.rbm.h_sample is never used, why is that here?
            # Moreover, it does not make sense for things like ssRBM.
            self.rbm.h_sample: _locals['h_mean']
        }