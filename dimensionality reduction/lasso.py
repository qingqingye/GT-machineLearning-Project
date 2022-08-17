from sklearn import linear_model
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import genfromtxt


def normalize(data, target):
    """
    Given input data and input target, return the normalize vectors
    """
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    target = (target - np.mean(target)) / np.std(target)
    return data,target


def lars(data, target, eps=np.finfo(float).eps):
    """
    Lars algorithm, given input data and target, return the list of
    lars coefficients and correlations
    """
    # variable initialization
    target = target[:, np.newaxis]
    (n, p) = np.shape(data)
    m = min(p, n - 1)
    beta = np.zeros((m, p))
    mu = np.zeros((n, 1))
    gamma = np.zeros(m)
    correlations = np.zeros((m, p))
    active_set = np.array([])
    n_vars = 0
    is_correct_sign = 1
    i = 0
    mu_old = np.zeros((n, 1))

    # lars algorithm
    while n_vars < m:

        # vector of current correlations
        current_correlation = np.dot(data.T, target - mu)
        correlations[i, :] = current_correlation.T[0]

        # greatest absolute current correlations
        greatest_correlation = np.amax(np.abs(current_correlation))

        # we reach the required precision, exit
        if greatest_correlation < eps:
            break
        if i == 0:
            variable_to_add = (greatest_correlation == abs(current_correlation)).argmax()
        if is_correct_sign:
            active_set = np.append(active_set, variable_to_add).astype(int)
            n_vars = n_vars + 1

        # update state variables
        sign_of_active_set = np.sign(current_correlation[active_set])
        inactive_set = np.setdiff1d(np.arange(0, p), active_set)
        n_zeros = len(inactive_set)
        data_active_set = np.copy(data[:, active_set])

        # Calculate the equiangular vector
        matrix_g = np.dot(data_active_set.T, data_active_set)
        inverse_matrix_g = np.linalg.inv(matrix_g)
        matrix_a = 1 / np.sqrt(np.dot(np.dot(sign_of_active_set.T, inverse_matrix_g), sign_of_active_set))
        length_of_equiangular_vector = np.dot(matrix_a * inverse_matrix_g, sign_of_active_set)
        equiangular_vector = np.dot(data_active_set, length_of_equiangular_vector)

        # calculate the projection along the equiangular vector
        a = np.dot(data.T, equiangular_vector)
        beta_tmp = np.zeros((p, 1))

        # update gammas, the quantity that we need to move along the equiangular vector
        gamma_test = np.zeros((n_zeros, 2))
        if n_vars == m:
            gamma[i] = greatest_correlation / matrix_a
        else:
            for j in range(n_zeros):
                jj = inactive_set[j]
                gamma_test[j, :] = np.concatenate(((greatest_correlation - current_correlation[jj]) / (matrix_a
                                                                                                       - a[jj]), (greatest_correlation + current_correlation[jj]) / (matrix_a + a[jj])), axis=1)
            (gamma[i], min_i, min_j) = min_positive(gamma_test)
            variable_to_add = np.unique(inactive_set[int(min_i)])
        beta_tmp[active_set] = beta[i - 1, active_set].reshape(beta_tmp[active_set].shape) \
                               + np.dot(gamma[i], length_of_equiangular_vector).reshape(beta_tmp[active_set].shape)

        # update mu doing the step mu = mu_old + gamma * equiangular_vector
        mu = mu_old + np.dot(gamma[i], equiangular_vector)
        mu_old = np.copy(mu)
        beta[i, :] = beta_tmp.T[0]
        i = i + 1
    return beta.T, correlations


def min_positive(a):
    """
    Find the minimum of the positive elements in an array a
    """
    a[a.imag != 0] = np.finfo(float).max
    a[a <= 0] = np.finfo(float).max
    size = a.shape[1]
    i = np.argmin(a) / size
    j = np.argmin(a) % size
    return (np.min(a), i, j)


def plot_coefficients(coefs):
    xx = np.sum(np.abs(coefs.T), axis=1)
    xx /= xx[-1]
    plt.plot(xx, coefs.T)
    ymin, ymax = plt.ylim()
    plt.vlines(xx, ymin, ymax, linestyle='dashed')
    plt.xlabel('|coef| / max|coef|')
    plt.ylabel('Coefficients')
    plt.title('coef vs s')
    plt.legend(['Housing Inventory', 'UnemploymentRate','cases','cases_rate', 'deaths','death_rate', 'fully_vaccinated','fully_vaccinated_rate', 'Population', 'Area','GDP', "GDPpp"])
    plt.axis('tight')
    plt.show()

def plot_coe_alpha(coefs, alpha):
    x = alpha
    y = coefs.T
    plt.plot(x, y)
    ymin, ymax = plt.ylim()
    plt.vlines(x, ymin, ymax, linestyle='dashed')
    plt.title('coef vs alpha')

    plt.legend(['Housing Inventory', 'UnemploymentRate','cases','cases_rate', 'deaths','death_rate', 'fully_vaccinated','fully_vaccinated_rate', 'Population', 'Area', 'GDP', "GDPpp"])

    plt.show()



def import_and_normalize_diabetes():
    # import file
    fp_df = pd.read_csv("df_after_FINAL_withgeoinfo.csv", thousands=',')    # feature price dataframe
    fp_df = fp_df.dropna()
    #x_df = fp_df.[["Housing Inventory UnemploymentRate","cases","deaths","cases_rate", "death_rate", "fully_vaccinated","fully_vaccinated_rate"]]
    x_df = fp_df.iloc[: ,6:].apply(pd.to_numeric)
    y_df = fp_df.iloc[:,5].apply(pd.to_numeric)

    x_var  = np.array(x_df)
    y_var = np.array(y_df)
    print(x_var.shape)
    print(y_var)
    x_var, y_var = normalize(x_var, y_var)

    return x_var, y_var


if __name__ == '__main__':

    #x_var, y_var = datasets.load_diabetes(return_X_y=True)
    x_var, y_var = import_and_normalize_diabetes()
    print("Computing regularization path using the LARS ...")

    # run lars
    coefs, correlations = lars(x_var, y_var)
    plot_coefficients(coefs=coefs)

    # run lars sklearn
    alphas, _, coefs = linear_model.lars_path(x_var, y_var, method='lasso', verbose=False)
    plot_coe_alpha(coefs, alphas)
    plot_coefficients(coefs=coefs)