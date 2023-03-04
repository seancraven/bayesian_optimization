import torch
from torch import nn
from l2distance import l2distance
from torch.nn import functional as F


def computeK(kernel_type, X, Z, kpar=0):
    """
    function K = computeK(kernel_type, X, Z)
    computes a matrix K such that Kij=k(x,z);
    for three different function linear, rbf or polynomial.

    Input:
    kernel_type: either 'linear','polynomial','rbf'
    X: n input vectors of dimension d (nxd);
    Z: m input vectors of dimension d (mxd);
    kpar: kernel parameter (inverse kernel width gamma in case of RBF,
    degree in case of polynomial)

    OUTPUT:
    K : nxm kernel Torch float tensor
    """
    assert kernel_type in ["linear", "polynomial", "poly",
                           "rbf"], "Kernel type %s not known." % kernel_type
    assert X.shape[1] == Z.shape[
        1], f"Input dimensions do not match X:{X.shape[1]} , Z:{Z.shape[1]}"
    lin = lambda x, z: torch.einsum("ij,kj -> ik", x, z)
    poly = lambda x, z: (torch.einsum("ij, kj -> ik", x, z) + 1) ** kpar
    rbf = lambda x, z: torch.exp(- kpar * l2distance(x, z) ** 2)
    kernel_dict = {"linear": lin, "polynomial": poly, "rbf": rbf, "poly": poly}
    K = kernel_dict[kernel_type](X, Z)
    return K


class KernelizedSVM(nn.Module):
    def __init__(self, dim, kernel_type, kpar=0):
        super().__init__()
        self.kernel_type = kernel_type
        self.dim = dim
        self.beta = nn.Parameter(torch.rand(dim), requires_grad=True)
        self.b = nn.Parameter(torch.rand(1), requires_grad=True)
        self.kpar = kpar

    def make_k(self, xTr):
        return computeK(self.kernel_type, xTr, xTr, self.kpar)

    def forward(self, xTr, x):
        K = computeK(self.kernel_type, xTr, x, self.kpar)
        return self.beta @ K + self.b


def kernelsvm_loss(kernelizedSVM, kernel_mat, yTr, C):
    beta = kernelizedSVM.beta
    det = 1 / 2 * beta @ kernel_mat @ beta
    pred = beta @ kernel_mat + kernelizedSVM.b
    assert pred.shape[0] == yTr.shape[0]
    hinge = C / 2 + (F.relu(1 - yTr * pred) ** 2).sum()

    return det + hinge

def dualSVM(xTr, yTr, kernel_type, num_epochs=100, C=1, lmbda=0, lr=1e-3):
    ksvm = KernelizedSVM(yTr.shape[0], kernel_type, lmbda)
    opt = torch.optim.SGD(ksvm.parameters(), lr=lr)
    k_mat = ksvm.make_k(xTr)
    for epoch in range(num_epochs):
        opt.zero_grad()
        loss = kernelsvm_loss(ksvm, k_mat, yTr, C)
        loss.backward()
        opt.step()
    return lambda x: ksvm(xTr, x)


def cross_validation(xTr, yTr, xValid, yValid, ktype, CList, lmbdaList, lr_List
                     ):
    """
    function bestC,bestLmbda,ErrorMatrix = cross_validation(xTr,yTr,xValid,yValid,ktype,CList,lmbdaList);
    Use the parameter search to find the optimal parameter,
    Individual models are trained on (xTr,yTr) while validated on (xValid,yValid)

    Input:
        xTr      | training data (nxd)
        yTr      | training labels (nx1)
        xValid   | training data (mxd)
        yValid   | training labels (mx1)
        ktype    | the type of kernelization: 'rbf','polynomial','linear'
        CList    | The list of values to try for the SVM regularization parameter C (ax1)
        lmbdaList| The list of values to try for the kernel parameter lmbda- degree for poly, inverse width for rbf (bx1)
        lr_list  | The list of values to try for the learning rate of our optimizer

    Output:
        bestC      | the best C parameter
        bestLmbda  | the best Lmbda parameter
        bestLr     | the best Lr parameter
        ErrorMatrix| the test error rate for each given (C, Lmbda Lr) tuple when trained on (xTr,yTr) and tested on (xValid,yValid)
    """

    best_loss = torch.inf
    ErrorMatrix = torch.ones((len(CList), len(lmbdaList), len(lr_List)),
                             requires_grad=False
                             )
    for i, C in enumerate(CList):
        for j, lmbda in enumerate(lmbdaList):
            for k, lr in enumerate(lr_List):
                ksvm = dualSVM(xTr, yTr, ktype, num_epochs=1000, C=C,
                               lmbda=lmbda, lr=lr
                               )
                train_error = (torch.sign(ksvm(xTr)) != yTr).float().mean()
                error_rate = (torch.sign(ksvm(xValid)) != yValid).float().mean()
                ErrorMatrix[i, j, k] = error_rate
                if error_rate < best_loss:
                    best_loss = error_rate
                    # print(error_rate)
                    # print(train_error)
                    bestC = C
                    bestLmbda = lmbda
                    bestLr = lr
    return bestC, bestLmbda, bestLr, ErrorMatrix


def autosvm(xTr, yTr):
    """
    svmclassify = autosvm(xTr,yTr), where yTe = svmclassify(xTe)
    """
    # keep sample balance in train val split.
    pos_index = torch.argwhere((yTr == 1).float())
    neg_index = torch.argwhere((yTr != 1).float())

    x_pos, y_pos = xTr[pos_index], yTr[pos_index]
    x_neg, y_neg = xTr[neg_index], yTr[neg_index]

    pos_val_no = pos_index.shape[0] // 5
    neg_val_no = neg_index.shape[0] // 5

    pos_ind = torch.randperm(pos_index.shape[0])
    neg_ind = torch.randperm(neg_index.shape[0])

    pos_train_ind, pos_val_ind = pos_ind[:-pos_val_no], pos_ind[-pos_val_no:]
    x_pos_train, y_pos_train = x_pos[pos_train_ind].squeeze(), y_pos[
        pos_train_ind].squeeze()
    x_pos_val, y_pos_val = x_pos[pos_val_ind].squeeze(), y_pos[
        pos_val_ind].squeeze()

    neg_train_ind, neg_val_ind = neg_ind[:-neg_val_no].squeeze(), neg_ind[
                                                                  -neg_val_no:]
    x_neg_train, y_neg_train = x_neg[neg_train_ind].squeeze(), y_neg[
        neg_train_ind].squeeze()
    x_neg_val, y_neg_val = x_neg[neg_val_ind].squeeze(), y_neg[
        neg_val_ind].squeeze()

    x_train = torch.cat([x_pos_train, x_neg_train], dim=0)
    y_train = torch.cat([y_pos_train, y_neg_train], dim=0)
    x_val = torch.cat([x_pos_val, x_neg_val], dim=0)
    y_val = torch.cat([y_pos_val, y_neg_val], dim=0)

    # Basic log grid search, depending on accuracy can do linear search inside min log values.
    CList = (np.logspace(0, 4, 10))
    lmbdaList = (np.logspace(-1, 4, 10))
    lrList = ([0.001])

    c_star, lm_star, lr_star, error_mat = cross_validation_adv(x_train, y_train,
                                                               x_val, y_val,
                                                               "rbf", CList,
                                                               lmbdaList,
                                                               lrList, mom=0.9,
                                                               n_epoch=70
                                                               )
    # best_svm = dualSVM(xTr, yTr, "rbf", 100,c_star, lm_star, lr_star, 0.9)
    print("-----------------------------------------------------------")
    print(f"Origional C range: [{float(CList[0])}, {float(CList[-1])}")
    print(f"Origional lm range: [{float(lmbdaList[0])}, {float(lmbdaList[-1])}")
    print(error_mat.squeeze())

    c_ind = np.argwhere(CList == c_star)
    lm_ind = np.argwhere(lmbdaList == lm_star)
    c_low, c_high = max(c_ind - 1, 0), min(c_ind + 1, len(CList) - 1)
    lm_low, lm_high = max(lm_ind - 1, 0), min(lm_ind + 1, len(lmbdaList) - 1)

    n_Clist = np.linspace(float(CList[c_low]), float(CList[c_high]), 10)
    n_lmlist = np.linspace(float(lmbdaList[lm_low]), float(lmbdaList[lm_high]),
                           10
                           )

    c_star, lm_star, lr_star, error_mat = cross_validation_adv(x_train, y_train,
                                                               x_val, y_val,
                                                               "rbf", n_Clist,
                                                               n_lmlist, lrList,
                                                               mom=0.9,
                                                               n_epoch=100
                                                               )
    best_svm = dualSVM_adv(xTr, yTr, "rbf", 100, c_star, lm_star, lr_star, 0.9,
                           clip=True
                           )
    print("-----------------------------------------------------------")
    print(f"C range: [{float(CList[c_low])}, {float(CList[c_high])}")
    print(f"lm range: [{float(lmbdaList[lm_low])}, {float(lmbdaList[lm_high])}")
    print(error_mat.squeeze())

    print(f"C Star: {c_star}, Lmbda Star: {lm_star}")
    return lambda x: torch.sign(best_svm(x))


def dualSVM_adv(xTr, yTr, kernel_type, num_epochs=100, C=1, lmbda=0, lr=1e-3,
                mom=0.0, clip=False
                ):
    ksvm = KernelizedSVM(yTr.shape[0], kernel_type, lmbda)
    opt = torch.optim.SGD(ksvm.parameters(), lr=lr, momentum=mom)
    #opt = torch.optim.Adam(ksvm.parameters())
    k_mat = ksvm.make_k(xTr)
    for epoch in range(num_epochs):
        opt.zero_grad()
        loss = kernelsvm_loss(ksvm, k_mat, yTr, C)
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(ksvm.parameters(), 5e9)
        opt.step()
    return lambda x: ksvm(xTr, x)


def cross_validation_adv(xTr, yTr, xValid, yValid, ktype, CList, lmbdaList,
                         lr_List, mom=0.9, n_epoch=1000
                         ):
    """
    function bestC,bestLmbda,ErrorMatrix = cross_validation(xTr,yTr,xValid,yValid,ktype,CList,lmbdaList);
    Use the parameter search to find the optimal parameter,
    Individual models are trained on (xTr,yTr) while validated on (xValid,yValid)

    Input:
        xTr      | training data (nxd)
        yTr      | training labels (nx1)
        xValid   | training data (mxd)
        yValid   | training labels (mx1)
        ktype    | the type of kernelization: 'rbf','polynomial','linear'
        CList    | The list of values to try for the SVM regularization parameter C (ax1)
        lmbdaList| The list of values to try for the kernel parameter lmbda- degree for poly, inverse width for rbf (bx1)
        lr_list  | The list of values to try for the learning rate of our optimizer

    Output:
        bestC      | the best C parameter
        bestLmbda  | the best Lmbda parameter
        bestLr     | the best Lr parameter
        ErrorMatrix| the test error rate for each given (C, Lmbda Lr) tuple when trained on (xTr,yTr) and tested on (xValid,yValid)
    """
    best_loss = 1
    bestC = bestLmbda = bestLr = 0
    ErrorMatrix = torch.ones((len(CList), len(lmbdaList), len(lr_List)),
                             requires_grad=False
                             )
    for i, C in enumerate(CList):
        for j, lmbda in enumerate(lmbdaList):
            for k, lr in enumerate(lr_List):

                ksvm = dualSVM_adv(xTr, yTr, ktype, num_epochs=n_epoch, C=C,
                                   lmbda=lmbda, lr=lr, clip=False
                                   )
                train_error = (torch.sign(ksvm(xTr)) != yTr).float().mean()
                error_rate = (torch.sign(ksvm(xValid)) != yValid).float().mean()
                ErrorMatrix[i, j, k] = error_rate

                if error_rate < best_loss:
                    best_loss = error_rate
                    # print(error_rate)
                    # print(train_error)
                    bestC = C
                    bestLmbda = lmbda
                    bestLr = lr

    return bestC, bestLmbda, bestLr, ErrorMatrix
