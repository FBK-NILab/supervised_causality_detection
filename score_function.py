import numpy as np
from create_trainset import class_to_configuration


def score(configuration_true, configuration_pred, binary_score):
    """Score a single prediction of the confuguration matrix.
    """
    configuration_true = configuration_true.astype(np.bool)
    configuration_pred = configuration_pred.astype(np.bool)
    np.fill_diagonal(configuration_true, False)
    np.fill_diagonal(configuration_pred, False)
    score = ( binary_score[0] * (configuration_true[configuration_pred]).astype(np.int)).sum() + \
            ( binary_score[1] * (1.0 - configuration_pred[configuration_true]).astype(np.int)).sum() + \
            ( binary_score[2] * (1.0 - configuration_true[configuration_pred]).astype(np.int)).sum() + \
            ( binary_score[3] * (1.0 - configuration_true[np.logical_not(configuration_pred)]).astype(np.int)).sum() - (binary_score[3]*3) #remove the diagonal
    return float(score)


def compute_score_matrix(n=64, binary_score=[1,0,-3,0]): #binary_score=[true_pos, false_neg, false_pos, true_neg]
    score_matrix = np.zeros((n, n))
    for c_true in range(n):
        for c_pred in range(n):
            score_matrix[c_true, c_pred] = score(class_to_configuration(c_true, verbose=False), class_to_configuration(c_pred, verbose=False), binary_score)

    return score_matrix


def best_decision(prob_configuration, score_matrix=None):
    """Given the probability of each configuration, compute the
    expected scores and the best decision.
    """
    if score_matrix is None:
        score_matrix = compute_score_matrix()
    
    # Sanity checks:
    #assert(len(prob_configuration)==64)
    assert((prob_configuration >=0).all())
    prob_configuration = prob_configuration / prob_configuration.sum()

    # scores = np.zeros(64)
    # for c_decided in range(64):
    #     # score[c_decided] = \int_{c_true} p(c_true|X) score(c_true, c_decided) d c_true :
    #     for c_true in range(64):
    #         scores[c_decided] += prob_configuration[c_true] * score_matrix[c_true, c_decided]

    # This is the previous code in vectorized format:
    scores = (score_matrix * prob_configuration[:,None]).sum(0)

    best = scores.argmax()
    return best, scores
    

if __name__ == '__main__':

    np.random.seed(0)

    from create_trainset import class_to_configuration

    c_true = 1
    c_pred = 2
    configuration_true = class_to_configuration(c_true)
    configuration_pred = class_to_configuration(c_pred)

    print "Score:", score(configuration_true, configuration_pred)

    score_matrix = compute_score_matrix()

    # prob_configuration = np.random.rand(64)
    # prob_configuration /= prob_configuration.sum()
    # prob_configuration = np.random.dirichlet(alpha=np.ones(64))
    prob_configuration = np.random.dirichlet(alpha=np.arange(64)**2)

    print "Given", prob_configuration
    best, scores = best_decision(prob_configuration, score_matrix=score_matrix)
    print "The score of each decision is:", scores
    print "The best decision is:", best
    print "With score:", scores[best]
    print "And p(c|X):", prob_configuration[best]
