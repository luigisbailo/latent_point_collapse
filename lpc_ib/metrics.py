import warnings
import sys
import scipy
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax, digamma
from scipy.spatial import cKDTree, distance
import faiss


def get_entropy(
        evaluations,
        k,
        base=None
        ):
        
    x_penultimate = evaluations['x_penultimate']

    
    n_samples, n_dimensions = x_penultimate.shape

    tree = cKDTree(x_penultimate)

    distances, _ = tree.query(x_penultimate, k=k+1, p=np.inf) 
    distances = distances[:, -1] 

    epsilon = 1e-10
    distances = distances + epsilon

    avg_log_dist = np.mean(np.log(distances))

    if base is None:
        const = digamma(n_samples) - digamma(k) + n_dimensions * np.log(2)
        entropy = const + n_dimensions * avg_log_dist
    else:
        const = (digamma(n_samples) - digamma(k) + n_dimensions * np.log(2)) / np.log(base)
        entropy = const + (n_dimensions * avg_log_dist) / np.log(base)

    return entropy


def get_coeff_var(
        evaluations,
        ):
        
    x_penultimate = evaluations['x_penultimate']
            
    norm = np.linalg.norm(evals_train['x_penultimate'], axis=1)
    coeff_var = np.std(norm)/np.mean(norm)

    return coeff_var


def get_collapse_metrics(evaluations):
    """
    Calculate collapse metrics in the penultimate layer.

    Parameters:
    - evaluations (dict): A dictionary containing the evaluation results.
        - 'x_penultimate' (numpy.ndarray): The penultimate layer activations.
        - 'y_predicted' (numpy.ndarray): The predicted labels.
        - 'y_label' (numpy.ndarray): The true labels.

    Returns:
    - collapse_results_dict (dict): A dictionary containing the collapse metrics.
        - 'within_class_variation' (float): The within-class covariance \
            compared with the between class covariance.
        - 'equiangular' (float): The equiangularity metric.
        - 'maxangle' (float): The maximum angle metric.
        - 'equinorm' (float): The equinorm metric.
        - 'sigma_w' (float): The within-class covariance.
    """

    x_penultimate = evaluations['x_penultimate']
    y_predicted = evaluations['y_predicted']
    y_label = evaluations['y_label']

    collapse_results_dict = {}
    class_mean_centered = []
    sigma_w = []
    sigma_b = []

    global_mean = np.mean(x_penultimate, axis=0)

    for class_label in np.unique(y_label):

        selection = y_predicted == class_label
        if (np.sum(selection) > 0):

            class_mean = np.mean(x_penultimate[selection], axis=0)
            class_mean_centered.append(class_mean-global_mean)
            sigma_w.append(np.cov((x_penultimate[selection]).T, rowvar=True))
            sigma_b.append(np.outer(class_mean - global_mean, class_mean - global_mean))

    class_mean_centered = np.stack(class_mean_centered, axis=0)
    sigma_w = np.mean(np.stack(sigma_w, axis=0), axis=0)
    sigma_b = np.mean(np.stack(sigma_b, axis=0), axis=0)
    within_class_variation = np.trace(
        np.matmul(sigma_w, scipy.linalg.pinv(sigma_b))
        ) / len(class_mean_centered)
    within_class_variation_weighted = within_class_variation / x_penultimate.shape[1]
    
    sigma_w = np.mean(sigma_w)

    cosine = []
    cosine_max = []

    for i in range(0, len(class_mean_centered)):
        for j in range(i+1, len(class_mean_centered)):
            cosine_max.append(
                np.abs(
                    np.dot(class_mean_centered[i], class_mean_centered[j])
                    / np.linalg.norm(class_mean_centered[i])
                    / np.linalg.norm(class_mean_centered[j])
                    + 1./(len(class_mean_centered)-1))
                       )
            cosine.append(
                np.dot(class_mean_centered[i], class_mean_centered[j])
                / np.linalg.norm(class_mean_centered[i])
                / np.linalg.norm(class_mean_centered[j]))

    cosine = np.stack(cosine)

    
    equiangular = np.std(cosine)
    maxangle = np.mean(cosine_max)
    equinorm = np.std(np.linalg.norm(class_mean_centered, axis=1))\
        / np.mean(np.linalg.norm(class_mean_centered, axis=1))    
    
    collapse_results_dict['within_class_variation'] = within_class_variation
    collapse_results_dict['within_class_variation_weighted'] = within_class_variation_weighted
    collapse_results_dict['equiangular'] = equiangular
    collapse_results_dict['maxangle'] = maxangle
    collapse_results_dict['equinorm'] = equinorm
    collapse_results_dict['sigma_w'] = sigma_w

    return collapse_results_dict


def get_binarity_metrics(evaluations):
    """
    Calculate binarity metrics.

    Args:
        evaluations (dict): A dictionary containing the evaluations data.
            - 'x_penultimate' (numpy.ndarray): The penultimate layer activations.


    Returns:
        dict: A dictionary containing the binarity metrics results, including:
            - 'score': The mean score of the Gaussian Mixture Models.
            - 'stds': The mean standard deviations of the Gaussian Mixture Models.
            - 'peaks_distance_mean': The mean distance between the peaks of the Gaussian Mixture Models.
            - 'peaks_distance_std': The standard deviation of the distances \
                between the peaks of the Gaussian Mixture Models.
    """

    x_penultimate = evaluations['x_penultimate']

    binarity_results_dict = {}
    score = []
    stds = []
    peaks_distance = []

    for d in range(x_penultimate.shape[1]):

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gmm = GaussianMixture(n_components=2)
                scaler = StandardScaler()
                scaler.fit(x_penultimate[:, d].reshape(-1, 1))
                x_penultimate_scaled = scaler.transform(
                    x_penultimate[:, d].reshape(-1, 1)
                    )
                gmm.fit(x_penultimate_scaled)
                score.append(gmm.score(x_penultimate_scaled))
                means = gmm.means_.flatten()
                std = np.sqrt(gmm.covariances_).flatten()
                stds.append(std)
                peaks_distance.append(np.abs(means[0]-means[1])/np.mean(std))

        except Exception as e:
            print('Error in GMM fit:', str(e))
            sys.exit(1)

    score = np.mean(score)
    stds = np.mean(stds)
    peaks_distance_mean = np.mean(peaks_distance)
    peaks_distance_std = np.std(peaks_distance)

    binarity_results_dict['score'] = score
    binarity_results_dict['stds'] = stds
    binarity_results_dict['peaks_distance_mean'] = peaks_distance_mean
    binarity_results_dict['peaks_distance_std'] = peaks_distance_std

    return binarity_results_dict


def mahalanobis_distance(points, mean, cov):
    """
    Calculate the Mahalanobis distance between each point and the mean.

    Parameters:
    - points (array-like): The points for which to calculate the distance.
    - mean (array-like): The mean point.
    - cov (array-like): The covariance matrix.

    Returns:
    - distances (ndarray): The Mahalanobis distances between each point and the mean.
    """

    inv_cov = np.linalg.inv(cov)
    distances = [distance.mahalanobis(point, mean, inv_cov) for point in points]

    return np.stack(distances)


def get_mahalanobis_score(
        evaluations_train,
        evaluations_test
        ):
    """
    Calculate auroc value based on the Mahalanobis score.

    Parameters:
    evaluations_train (dict): Dictionary containing evaluation results for the training set.
        - 'x_penultimate' (numpy.ndarray): Array of penultimate layer features for the training set.
        - 'y_predicted' (numpy.ndarray): Array of predicted labels for the training set.
        - 'y_label' (numpy.ndarray): Array of true labels for the training set.
    evaluations_test (dict): Dictionary containing evaluation results for the test set.
        - 'x_penultimate' (numpy.ndarray): Array of penultimate layer features for the test set.
        - 'y_predicted' (numpy.ndarray): Array of predicted labels for the test set.
        - 'y_label' (numpy.ndarray): Array of true labels for the test set.

    Returns:
    dict: Dictionary containing the Mahalanobis score and other evaluation metrics.
        - 'score_classified_train' (numpy.ndarray): Array of Mahalanobis \
            scores for correctly classified instances in the training set.
        - 'score_classified_test' (numpy.ndarray): Array of Mahalanobis \
            scores for correctly classified instances in the test set.
        - 'score_misclassified_train' (numpy.ndarray): Array of Mahalanobis \
            scores for misclassified instances in the training set.
        - 'score_misclassified_test' (numpy.ndarray): Array of Mahalanobis \
            scores for misclassified instances in the test set.
        - 'auroc' (float): Area under the receiver operating characteristic \
            curve (AUROC) score.
    """

    score = {}
    score_classified_train = np.array([])
    score_classified_test = np.array([])
    score_misclassified_train = np.array([])
    score_misclassified_test = np.array([])

    x_penultimate_train = evaluations_train['x_penultimate']
    y_predicted_train = evaluations_train['y_predicted']
    y_label_train = evaluations_train['y_label']

    x_penultimate_test = evaluations_test['x_penultimate']
    y_predicted_test = evaluations_test['y_predicted']
    y_label_test = evaluations_test['y_label']

    scaler = StandardScaler()
    scaler.fit(x_penultimate_train)
    x_penultimate_train_scaled = scaler.transform(x_penultimate_train)
    x_penultimate_test_scaled = scaler.transform(x_penultimate_test)

    for class_label in np.unique(y_label_train):

        selection_classified_train = np.logical_and(
            y_predicted_train == class_label,
            y_label_train == y_predicted_train
            )
        selection_misclassified_train = np.logical_and(
            y_predicted_train == class_label,
            y_label_train != y_predicted_train
            )
        selection_classified_test = np.logical_and(
            y_predicted_test == class_label,
            y_label_test == y_predicted_test
            )
        selection_misclassified_test = np.logical_and(
            y_predicted_test == class_label,
            y_label_test != y_predicted_test
            )

        if np.sum(selection_classified_train) > 2:

            gmm = GaussianMixture(n_components=1)
            gmm.fit(x_penultimate_train_scaled[selection_classified_train])
            means = gmm.means_[0]
            covariances = gmm.covariances_[0]

            score_classified_train = np.concatenate([
                score_classified_train,
                mahalanobis_distance(
                    x_penultimate_train_scaled[selection_classified_train],
                    means,
                    covariances)
                    ])

            if np.sum(selection_classified_test) > 2:
                score_classified_test = np.concatenate([
                    score_classified_test,
                    mahalanobis_distance(
                        x_penultimate_test_scaled[selection_classified_test],
                        means,
                        covariances)
                        ])

            if np.sum(selection_misclassified_train) > 2:
                score_misclassified_train = np.concatenate([
                    score_misclassified_train,
                    mahalanobis_distance(
                        x_penultimate_train_scaled[selection_misclassified_train],
                        means,
                        covariances)
                        ])

            if np.sum(selection_misclassified_test) > 2:
                score_misclassified_test = np.concatenate([
                    score_misclassified_test,
                    mahalanobis_distance(
                        x_penultimate_test_scaled[selection_misclassified_test],
                        means,
                        covariances)
                        ])

    dx = 0.01
    x_axis = np.arange(0, 1+dx, dx)
    auroc = 0
    for x in x_axis:
        q = np.quantile(score_misclassified_test, x)
        y = np.mean(score_classified_test < q)
        auroc += y*dx

    score['score_classified_train'] = score_classified_train
    score['score_classified_test'] = score_classified_test
    score['score_misclassified_train'] = score_misclassified_train
    score['score_misclassified_test'] = score_misclassified_test
    score['auroc'] = auroc

    return score

def get_odin_score(
        evaluations_train,
        evaluations_test,
        evaluations_ood_list,
        T
        ):
    
    def get_score(x,T):
        
        exp = np.exp(softmax(x, axis=1)/T)
        score = np.max(exp, axis=1) / np.sum(exp, axis=1)
        
        return score
        
    score = {}
    
    x_output_train = evaluations_train['x_output']
    y_predicted_train = evaluations_train['y_predicted']
    y_label_train = evaluations_train['y_label']

    x_output_test = evaluations_test['x_output']
    y_predicted_test = evaluations_test['y_predicted']
    y_label_test = evaluations_test['y_label']

    x_output_ood = []
    y_predicted_ood = []
    y_label_ood = []

    for i in range(len(evaluations_ood_list)):
        x_output_ood.append(evaluations_ood_list[i]['x_output'])
        y_predicted_ood.append(evaluations_ood_list[i]['y_predicted'])
        y_label_ood.append(evaluations_ood_list[i]['y_label'])    

    selection_classified_train = y_label_train == y_predicted_train
    selection_misclassified_train = y_label_train != y_predicted_train
    selection_classified_test = y_label_test == y_predicted_test
    selection_misclassified_test = y_label_test != y_predicted_test
    
    score_classified_train = get_score (x_output_train[selection_classified_train],T)
    score_misclassified_train = get_score(x_output_train[selection_misclassified_train],T)
    score_classified_test = get_score(x_output_test[selection_classified_test],T)
    score_misclassified_test = get_score(x_output_test[selection_misclassified_test],T)
    score_test = get_score(x_output_test,T)

    score_ood = []
    for i in range (len(evaluations_ood_list)):
        score_ood.append(get_score(x_output_ood[i]/T))

    dx = 0.01
    x_axis = np.arange(0, 1+dx, dx)
    
    auroc = 0
    for x in x_axis:
        q = np.quantile(score_misclassified_test,1 - x)
        y = np.mean(score_classified_test > q)
        auroc += y*dx
    auroc = auroc
    
    auroc_ood = []
    for i in range (len(evaluations_ood_list)):
        auroc_ood.append(0)
        for x in x_axis:
            q = np.quantile(score_ood[i],1 - x)
            y = np.mean(score_test > q)
            auroc_ood[i] += y*dx
        auroc_ood[i] = auroc_ood[i]
   
    score['score_ood'] = score_ood    
    score['score_classified_train'] = score_classified_train
    score['score_classified_test'] = score_classified_test
    score['score_misclassified_train'] = score_misclassified_train
    score['score_misclassified_test'] = score_misclassified_test
    score['auroc'] = auroc
    score['auroc_ood'] = auroc_ood
    
    return score
