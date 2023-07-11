'''
Disentanglement functionalities
'''

import numpy as np


def random_samples(x, y, n=100000):

    if n > x.shape[0]:
        n= x.shape[0]
    index = np.random.choice(x.shape[0], n, replace=False)
    return  x[index], y[index]


def distance_diagonal(cov, idx):
    '''

    '''

    perfect_cov = np.ones(cov.shape)
    perfect_cov[idx] = 0.0  # assign 0.0 to the dimension at idx

    return 1 - np.absolute(np.subtract(perfect_cov, np.clip(cov, 0., 1.))).mean()  # or sum

def order_magnitude(numbers):
    if np.isclose(numbers, 0.0):
        return np.zeros(numbers.shape)

    numbers = np.absolute(numbers)
    return np.floor(np.log10(numbers))


def check_overlap_factors(dict_association):
    dict_overlap = {}

    for factor_i, latents_i in dict_association.items():

        overlap = 0
        for factor_j, latents_j in dict_association.items():

            if factor_j == factor_i:
                continue

            overlap += len(set(latents_i) & set(latents_j))
        dict_overlap[factor_i] = overlap
    return dict_overlap


# extension: base score fo each factor is proportional how much the cov are [1,1,1,0,1], Distance * score
def disentanglement_score(dict_association, dict_cov, alpha=0.25):
    '''

    distance_weight: equal to zero means only if factor is found matters
    '''

    # check if duplicates
    dict_overlap = check_overlap_factors(dict_association)

    dict_score = {}  # assign all the perfect association 1.0
    for factor, latents in dict_association.items():

        dict_score[factor] = 0.0

        # factor not found in the representation
        if len(latents) <= 0:
            continue

        score = 1.0  # perfect score

        # multiple encoding penalization
        score -= (score - score / len(latents))

        # check factor overlap
        if dict_overlap[factor] > 0:
            score -= (score - score / (dict_overlap[factor] + 1))

        # distance to perfect disentanglement
        cov_distance = np.mean([distance_diagonal(dict_cov[factor], l) for l in latents])

        # weighted score
        dict_score[factor] = (1 - alpha) * score + alpha * cov_distance

    return np.mean(list(dict_score.values())).round(4), dict_score  # average


def are_dead_dimensions(covs, factors, threshold=1e-03):
    '''
    dict_cov [factor](latent_dim, latent_dim): covariance matrix of X, Y interving of one factor

    If cov is very low for each factor, then it is dead.
    return list of bool
    '''

    n_features, n_factors = covs.shape

    is_dead = [True for _ in range(n_features)]  # by default are all dead

    for i, factor in enumerate(factors):
        diag_cov = covs[:, i]

        almost_zeros = np.isclose(diag_cov, np.zeros(diag_cov.shape), rtol=0.0, atol=threshold)  # abs(a) <= atol

        is_dead = np.logical_and(is_dead, almost_zeros)

    return is_dead


def associate_dims_factors(covs, factors, dead_dims, threshold=0.4):
    '''

    '''

    n_features, n_factors = covs.shape

    dict_association = {}  # factor --> list of dims

    dict_cov = {}

    aux_covs = covs.copy()
    # first remove dead dims
    aux_covs[dead_dims] = np.full(n_factors, np.inf)

    # first threshold 0.4, True if val < 0.4
    t_cond = aux_covs < threshold
    aux_covs[~t_cond] = np.inf

    # find minimum x factor, if any
    bests = np.min(aux_covs, axis=0)

    # for each factor find dimensions with same order of magnitude
    for i, factor in enumerate(factors):
        best = bests[i]  # best of factor

        aux_cov = covs[:, i].copy()
        dict_cov[factor] = aux_cov

        dict_association[factor] = []

        # factor with no dimensions survived
        if best == np.inf:
            continue

        # find other dims with same magnitude
        best_magn = order_magnitude(best)
        i_mins = [i for i, value in enumerate(aux_cov) if value < threshold and order_magnitude(value) == best_magn]

        dict_association[factor] = i_mins
    return dict_association, dict_cov


def cov_factor(X, Y):
    '''

    '''

    factors = X.files  # assuming X.files and Y.files corresponds
    num_points, n_features = X[factors[0]].shape

    # print(num_points, n_features)

    dict_cov = {}

    for factor in factors:
        x, y = X[factor], Y[factor]
        x, y = random_samples(x, y)

        cov = np.cov(x, y, rowvar=False, bias=False)  # sample covariance
        cov = cov[:n_features, n_features:]

        # interested only in the diagonal
        dict_cov[factor] = np.diag(cov.round(4))
    covs = np.hstack([np.expand_dims(cov, axis=1) for factor, cov in dict_cov.items()])

    return covs, factors


def representation_info(X, Y):
    '''
    Given X and Y representations to which an intevertion has been applied according to the factor classes.
    Get info about representation.
    '''

    covs, factors = cov_factor(X, Y)

    is_dead = are_dead_dimensions(covs, factors)

    print("Dead dimensions [0, N): ", [i for i, dead in enumerate(is_dead) if dead])

    dict_association, dict_cov = associate_dims_factors(covs, factors, is_dead)

    print("Association factor --> dimension(s):  ", dict_association)

    score = disentanglement_score(dict_association, dict_cov)

    print("Disentanglement score: {}".format(score))


def get_score(X, Y, alpha=0.25, threshold=0.4):
    covs, factors = cov_factor(X, Y)
    is_dead = are_dead_dimensions(covs, factors)
    dict_association, dict_cov = associate_dims_factors(covs, factors, is_dead, threshold)
    score, factors_score = disentanglement_score(dict_association, dict_cov, alpha)
    return score, [i for i, dead in enumerate(is_dead) if dead], dict_association, factors_score



