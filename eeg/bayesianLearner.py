
def bayesianHierarchicalInverter(seq, K=1, priors=None, noise_sensory=None, 
                                 noise_inference=None, noise_decision=None, square=False):
    '''
    Bayesian inversion of a transition probabilities of sequences. 
    Estimate transition frequencies based on estimate of frequencies of pairs, tierces, etc ... depending on the level of hierarchy (K). 
    The goal is to estimate P(X|Y) where X is a item of the sequence and Y can be :
    - X(t-1) if K=1, 
    - X(t-2)X(t-1) if K=2, 
    - X(t-3)X(t-2)X(t-1) if K=3, 
    - ...
    '''
    
    import numpy as np
    import scipy.signal
    
    # Get shape
    n_seq, n_cat = seq.shape

    # Add sensory noise
    if noise_sensory is not None:
        if isinstance(noise_sensory, np.ndarray):
            kernel = noise_sensory
        else:
            import scipy.signal
            gaussian_kernel = scipy.signal.gaussian(M=n_cat*2-1, std=noise_sensory)
            convolve_vect = np.vectorize(np.convolve, signature='(m),(n),()->(m)')
            kernel = convolve_vect(np.identity(n_cat), gaussian_kernel, mode='valid')
        P = kernel[np.nonzero(seq)[1]]
        P /= np.max(P, axis=-1, keepdims=True)
        seq = P.copy()

    # Normalize input
    seq /= np.sum(seq, 1, keepdims=True)
            
    # Create Y to compute P(X|Y)
    # Y is unique if K=1, pairs if K=2, tierces if K=3, etc ...

    # Tricky manipulation :
    # if we want the pair at time i, we perform an inner product of 
    # sequence seq[i] and seq[i-1].
    # We do the same here but for the general case k-uplets. 
    time_shifted = []
    if square == False:
        kuplet = K+1
    else:
        if K != 0:
            kuplet = K*2
        else:
            kuplet = 1
    for i in np.arange(kuplet):

        # Shift array in time (pairs +1, tierces +2 and +1, etc ...)
        X = seq[i:len(seq)-(kuplet-1-i)]

        # Add dimensions to duplicate arrays
        shape = [1]
        for j in np.arange(kuplet-1):
            X = X[..., None]
            shape += [n_cat]

        # Get the proper transposition shape to have :
        # First dimension = first element of the k-uplets
        # Second dimension = second element of the k-uplets
        # ...
        transpose_shape = [0]
        for j in np.arange(kuplet):
            transpose_shape += [((j+i)%(kuplet))+1]

        # Duplicate arrays in all dimensions   
        X = np.tile(X, shape).transpose(transpose_shape)
        time_shifted += [X]

    # Perform inner product
    Y = np.ones(X.shape)
    for i in np.arange(kuplet):
        Y = Y * time_shifted[i]

    # Add uninformative priors
    if K != 0:
        if priors is None:
            priors_shape = [kuplet-1]
            for i in np.arange(kuplet):
                priors_shape += [n_cat]
            priors = np.ones((priors_shape))
            priors /= np.sum(priors)
        Y = np.concatenate([priors, Y], 0)
    else:
        if priors is None:
            priors = np.ones((n_cat))
            priors /= np.sum(priors)
        Y[0] += priors

    # Add memory decay
    if noise_inference is not None:
        times = np.arange(-n_seq/2., n_seq/2., dtype=float)[::-1]
        decay = np.exp(-times/noise_inference)
        shape = [1]
        for k in np.arange(K+1):
            shape.append(1)
        Y *= np.tile(decay, shape).T
        
    # Evolution of posterior predictive in time
    p_obs = np.cumsum(Y, 0)
    
    # Flatten array of p_obs for diplay purpose
    if square == False:
        p_obs = p_obs.reshape(n_seq, n_cat, -1, order='F').reshape(n_seq, -1, n_cat)
    else:
        p_obs = np.transpose(p_obs, (transpose_shape))
        p_obs_square = p_obs.reshape((n_seq, n_cat**K, -1), order='F')
        p_obs_square /= np.sum(p_obs_square, (2), keepdims=True)
        p_obs = p_obs_square.reshape((n_seq, n_cat**K, n_cat, -1), order='F')
        p_obs = np.sum(p_obs, axis=2)
        p_obs /= np.sum(p_obs, (2), keepdims=True)
        return p_obs, p_obs_square

    # Normalize posterior distribution
    p_obs /= np.sum(p_obs, (2), keepdims=True)

    # Marginal posterior across groups 
    # e.g. if we have strong value for AAB and ABB, 
    # even if we have AB or AA we want p(B) to be high
    p_obs_marg = np.empty((n_seq, n_cat))
    p_obs_marg[0] = np.sum(p_obs[0], axis=0) # First item based only on priors

    # Get context
    if K == 0:
        context = np.ones((n_seq)) # No context if K=0
        p_obs_marg[0] = 1./n_cat
    else:
        Y_trans = np.reshape(Y, (n_seq, n_cat, -1), order='F').reshape(n_seq, -1, n_cat)
        context = np.sum(Y_trans, axis=-1)
    context = context[..., None]

    # Compute marginal posterior across contexts
    p_obs_marg[1:] = np.sum(context[1:] * p_obs[:-1], axis=1)
    p_obs_marg /= np.sum(p_obs_marg, 1, keepdims=True)

    # Choice of the model (softmax)
    choice = np.zeros(n_seq)
    if noise_decision is None:
        choice = np.argmax(p_obs_marg, 1) # Perfect policy
    else:
        for i in np.arange(n_seq):
            p_obs_marg[i] = np.exp(p_obs_marg[i]/noise_decision)/np.sum(np.exp(p_obs_marg[i]/noise_decision))
            choice[i] = np.where(np.random.multinomial(1, p_obs_marg[i]))[0][0]

    # Surprise is -log(p)
    p = p_obs_marg[np.arange(n_seq), np.argmax(seq, 1)]
    surprise = -np.log(p)
    entropy = -p * np.log(p)
    
    # Return
    return p_obs_marg, p_obs, choice, surprise, entropy

