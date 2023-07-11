import numpy as np
from scipy.stats import norm


def european_option_pricing_explicit(S, K, r, q, sigma, T, N, M):
    """
    使用显式差分法计算欧式期权的定价

    参数:
    S: 标的资产价格
    K: 期权行权价
    r: 无风险利率
    q: 标的资产波动率
    sigma: 标的资产波动率
    T: 期权到期时间
    N: 价格网格步数
    M: 时间网格步数

    返回:
    option_price: 欧式期权的定价
    """

    # 计算网格步长
    delta_t = T / M
    delta_S = S / N

    # # 创建时间网格
    # t_values = np.linspace(0, T, M + 1)
    # # 创建价格网格
    S_values = np.linspace(0, S, N + 1)

    # 创建网格矩阵并初始化为0 该矩阵为定价矩阵,即结果
    option_price_explicit = np.zeros((M + 1, N + 1))

    # 使用显式差分法进行离散化和求解 A为迭代系数矩阵
    A = np.zeros((N - 1, N - 1))

    option_price_explicit[M, :] = np.maximum(S_values - K, 0)  # 到期日的支付
    option_price_explicit[:, 0] = 0
    for i in range(0, M+1):
        option_price_explicit[i, N] = S - K * np.exp(-r * (T - i * delta_t))

    A[0, 0] = (1 - delta_t * sigma ** 2 + +r * delta_t)
    A[0, 1] = delta_t * (0.5 * sigma ** 2 + 0.5 * (r - q))
    A[N-2, N-3] = delta_t * (0.5 * (sigma * (N-1)) ** 2 - 0.5 * (r - q) * (N-1))
    A[N-2, N-2] = (1 - delta_t * (sigma * (N-1)) ** 2 + r*delta_t)

    for j in range(1, N-2):
        A[j, j - 1] = delta_t * (0.5 * (sigma * (j+1)) ** 2 - 0.5 * (r - q) * (j+1))
        A[j, j] = (1 - delta_t * (sigma * (j+1)) ** 2 + r * delta_t)
        A[j, j + 1] = delta_t * (0.5 * (sigma * (j+1)) ** 2 + 0.5 * (r - q) * (j+1))

    CN_1 = delta_t * (0.5 * (sigma * (N-1)) ** 2 + 0.5 * (r - q) * (N-1))

    for i in range(M, 0, -1):
        V = option_price_explicit[i, 1: N]
        option_price_explicit[i - 1, 1: N] = np.dot(A, V)
        option_price_explicit[i-1, N-1] = option_price_explicit[i-1, N-1] + CN_1*option_price_explicit[i, N]

    def spectral_radius(M):
        a, b = np.linalg.eig(M)  # a为特征值集合，b为特征值向量
        return np.max(np.abs(a))  # 返回谱半径
    print(spectral_radius(A))
    return option_price_explicit


def european_option_pricing_implicit(S, K, r, q, sigma, T, N, M):
    """
    使用隐式差分法计算欧式期权的定价

    参数:
    S: 标的资产价格
    K: 期权行权价
    r: 无风险利率
    q: 标的资产波动率
    sigma: 标的资产波动率
    T: 期权到期时间
    N: 价格网格步数
    M: 时间网格步数

    返回:
    option_price: 欧式期权的定价
    """

    # 计算网格步长
    delta_t = T / M
    delta_S = S / N

    # # 创建时间网格
    # t_values = np.linspace(0, T, M + 1)
    # # 创建价格网格
    S_values = np.linspace(0, S, N + 1)

    # 创建网格矩阵并初始化为0
    option_price_implicit = np.zeros((M + 1, N + 1))
    option_price_implicit[M, :] = np.maximum(S_values - K, 0)  # 到期日的支付
    option_price_implicit[M, :] = np.maximum(S_values - K, 0)  # 到期日的支付
    option_price_implicit[:, 0] = 0
    for i in range(0, M+1):
        option_price_implicit[i, N] = S - K * np.exp(-r * (T - i * delta_t))

    # 使用隐式差分法进行离散化和求解 迭代系数矩阵
    A = np.zeros((N - 1, N - 1))

    A[0, 0] = 1 + delta_t * (sigma ** 2 + r)
    A[0, 1] = delta_t * (-0.5 * sigma ** 2 - 0.5 * (r - q))
    A[N - 2, N - 3] = delta_t * (-0.5 * (sigma * (N - 1)) ** 2 + 0.5 * (r - q) * (N-1))
    A[N - 2, N - 2] = delta_t * ((sigma * (N - 1)) ** 2 + r)

    for j in range(1, N-2):
        A[j, j - 1] = delta_t * (-0.5 * (sigma * (j+1)) ** 2 + 0.5 * (r - q) * (j+1))
        A[j, j] = 1 + delta_t * ((sigma * (j+1)) ** 2 + r)
        A[j, j + 1] = delta_t * (-0.5 * (sigma * (j+1)) ** 2 - 0.5 * (r - q) * (j+1))

    CN_1 = delta_t * (-0.5 * (sigma * (N - 1)) ** 2 - 0.5 * (r - q) * (N - 1))
    for i in range(M-1, 0, -1):
        V = option_price_implicit[i+1, 1:N]
        V[N-2] = V[N-2] - CN_1*option_price_implicit[i, N]
        option_price_implicit[i, 1:N] = np.dot(np.linalg.inv(A), V)

    return option_price_implicit


def european_option_pricing_CN(S, K, r, q, sigma, T, N, M, theta):
    """
    使用Crank-Nicholson差分法计算欧式期权的定价

    参数:
    S: 标的资产价格
    K: 期权行权价
    r: 无风险利率
    q: 标的资产波动率
    sigma: 标的资产波动率
    T: 期权到期时间
    N: 价格网格步数
    M: 时间网格步数

    返回:
    option_price: 欧式期权的定价
    """
    seta_a = theta
    seta_b = 1-theta
    # 计算网格步长
    delta_t = T / M
    delta_S = S / N

    # # 创建时间网格
    # t_values = np.linspace(0, T, M + 1)
    # # 创建价格网格
    S_values = np.linspace(0, S, N + 1)

    # 创建网格矩阵并初始化为0
    option_price_CN = np.zeros((M + 1, N + 1))
    option_price_CN[M, :] = np.maximum(S_values - K, 0)  # 到期日的支付
    option_price_CN[:, 0] = 0
    for i in range(0, M + 1):
        option_price_CN[i, N] = S - K * np.exp(-r * (T - i * delta_t))

    # 使用Crank-Nicholson差分法进行离散化和求解
    A = np.zeros((N - 1, N - 1))
    B = np.zeros((N - 1, N - 1))

    A[0,0] = 1 + seta_a * delta_t * (sigma ** 2 + r)
    A[0,1] =- seta_a * delta_t * ((sigma * 1) ** 2 / 2 + (r - q) * 1 / 2)
    A[N - 2, N - 3] = - seta_a * delta_t * ((sigma * (N-1)) ** 2 / 2 - (r - q) * (N-1) / 2)
    A[N - 2, N - 2] = 1 + seta_a * delta_t * ((sigma * (N - 1)) ** 2 + r)


    B[0, 0] =1 - seta_b * delta_t * ((sigma * 1) ** 2 + r)
    B[0, 1] = seta_b * delta_t * ((sigma * 1) ** 2 / 2 + (r - q) * 1 / 2)
    B[N - 2, N - 3] = seta_b * delta_t * ((sigma *(N-1)) ** 2 / 2 - (r - q) * (N-1)/ 2)
    B[N - 2, N - 2] =1 - seta_b * delta_t * ((sigma * (N-1)) ** 2 + r)

    for j in range(1, N-2):
        A[j, j - 1] = - seta_a * delta_t * ((sigma * j) ** 2 / 2 - (r - q) * j / 2)
        A[j, j] = 1 + seta_a * delta_t * ((sigma * j) ** 2 + r)
        A[j, j + 1] = - seta_a * delta_t * ((sigma * j) ** 2 / 2 + (r - q) * j / 2)

        B[j, j - 1] = seta_b * delta_t * ((sigma * j) ** 2 / 2 - (r - q) * j / 2)
        B[j, j] = 1 - seta_b * delta_t * ((sigma * j) ** 2 + r)
        B[j, j + 1] = seta_b * delta_t * ((sigma * j) ** 2 / 2 + (r - q) * j / 2)

    C = np.dot(np.linalg.inv(A), B)
    tempt = np.zeros((N - 1, 1))
    fn = seta_b * delta_t * ((sigma * (N - 1)) ** 2 / 2 + (r - q) * (N-1) / 2)
    cn1 = - seta_a * delta_t * ((sigma *(N-1)) ** 2 / 2 + (r - q) * (N-1) / 2)
    for i in range(M-1, 0, -1):
        tempt[N - 2, 0] = fn*option_price_CN[i+1, N]+cn1*option_price_CN[i, N]
        tempt = np.dot(np.linalg.inv(A), tempt)
        V = option_price_CN[i + 1, 1:N]
        option_price_CN[i, 1:N] = np.dot(C, V)
        option_price_CN[i, N-1] = option_price_CN[i, N-1]+tempt[N-2, 0]
    return option_price_CN


# 测试代码
S_test = 200  # 标的资产价格max
K_test = 113  # 期权行权价
r_test = 0.05/100  # 无风险利率
sigma_test = 0.4  # 标的资产波动率
q_test = 0  # 连续红利率
T_test = 1/365  # 期权到期时间
N_test = 500  # 价格网格步数
M_test = 500  # 时间网格步数

option_price_explicit_test = european_option_pricing_explicit(S_test, K_test, r_test, q_test, sigma_test, T_test, N_test,
                                                              M_test)
option_price_implicit_test = european_option_pricing_implicit(S_test, K_test, r_test, q_test, sigma_test, T_test,
                                                              N_test, M_test)
option_price_CN_test = european_option_pricing_CN(S_test, K_test, r_test, sigma_test, q_test, T_test,
                                                  N_test, M_test, 0.5)
'''option_price_CN_test_2 = european_option_pricing_CN(S_test, K_test, r_test, sigma_test, q_test, T_test,
                                                  N_test, M_test,0.3)
option_price_CN_test_3 = european_option_pricing_CN(S_test, K_test, r_test, sigma_test, q_test, T_test,
                                                  N_test, M_test,0.1)'''
print("欧式期权的定价（显式差分）:", option_price_explicit_test[0, 500])
print("欧式期权的定价（隐式差分）:", option_price_implicit_test[0, 500])  # 输出期权定价结果
print("欧式期权的定价（CN差分）:", option_price_CN_test[0, 500])
#print(np.shape(option_price_implicit_test))



#画图
from matplotlib import pyplot as plt
#作图

Y = np.arange(0, T_test+T_test/M_test, T_test/M_test)
X = np.arange(0, S_test+S_test/N_test, S_test/N_test)

'''#求理论值矩阵
theo_mat = np.zeros((M_test+1,N_test+1))
for i in range(0, M_test+1):
    for j in range(0, N_test+1):
        d1 = (np.log(X[j] / K_test) + (r_test + 0.5 * (sigma_test ** 2)) * T_test) / np.sqrt(Y[i]) / sigma_test
        d2 = (np.log(X[j] / K_test) + (r_test - 0.5 * (sigma_test ** 2)) * T_test) / np.sqrt(Y[i]) / sigma_test
        nd1 = norm.cdf(d1)
        nd2 = norm.cdf(d2)
        theo_mat[i, j] = S_test * nd1 - K_test * np.exp(-r_test * T_test) * nd2'''

X, Y = np.meshgrid(X, Y)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
cf = ax.plot_surface(X, Y, option_price_implicit_test, cmap=plt.get_cmap('rainbow'))
#cf = ax.plot_surface(X, Y, option_price_explicit_test, cmap=plt.get_cmap('rainbow'))
#cf=ax.plot_surface(X, Y, option_price_CN_test, cmap=plt.get_cmap('rainbow'))
#cf=ax.plot_surface(X, Y, theo_mat-option_price_explicit_test, cmap=plt.get_cmap('rainbow'))
#ax.plot_surface(X, Y, option_price_CN_test_1, cmap=plt.get_cmap('rainbow'))
#ax.plot_surface(X, Y, option_price_CN_test_2, cmap=plt.get_cmap('rainbow'))
#ax.plot_surface(X, Y, option_price_CN_test_3, cmap=plt.get_cmap('rainbow'))
plt.colorbar(cf)
plt.show()

#绘制时间error图
'''cn_error = np.arange(6)
im_error = np.arange(6)

cn_error = np.abs(option_price_CN_test_1[:, 500]-theo_mat[:, 500])
im_error = np.abs(option_price_implicit_test[:, 500]-theo_mat[:, 500])
print(cn_error)
print(im_error)
plt.plot(Y, cn_error, 'b--', label='C-N')  # 'bo-'表示蓝色实线，数据点实心原点标注
#plt.plot(Y, im_error, 'r--', label='Implicit')

plt.legend()  # 显示上面的label
plt.xlabel('time')  # x_label
plt.ylabel('error')  # y_label
plt.show()'''



#计算欧式看涨期权的理论值
d1 = (np.log(S_test / K_test) + (r_test + 0.5 * (sigma_test ** 2)) * T_test) / np.sqrt(T_test) / sigma_test
d2 = (np.log(S_test / K_test) + (r_test - 0.5 * (sigma_test ** 2)) * T_test) / np.sqrt(T_test) / sigma_test
nd1 = norm.cdf(d1)
nd2 = norm.cdf(d2)
TheoreticalResult = S_test * nd1 - K_test * np.exp(-r_test * T_test) * nd2
print("欧式期权的理论定价:", TheoreticalResult)
