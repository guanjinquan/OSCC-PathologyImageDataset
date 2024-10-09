# This code is from
# Multi-Task Learning as Multi-Objective Optimization
# Ozan Sener, Vladlen Koltun
# Neural Information Processing Systems (NeurIPS) 2018 
# https://github.com/intel-isl/MultiObjectiveOptimization
# replaced numpy with torch

import torch
import numpy as np


class MinNormSolver:
    MAX_ITER = 250
    STOP_CRIT = 1e-5

    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        """
        Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
        d is the distance (objective) optimzed
        v1v1 = <x1,x1>
        v1v2 = <x1,x2>
        v2v2 = <x2,x2>
        """
        if v1v2 >= v1v1:
            # Case: Fig 1, third column
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            # Case: Fig 1, first column
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        # Case: Fig 1, second column
        gamma = -1.0 * ( (v1v2 - v2v2) / (v1v1+v2v2 - 2*v1v2) )
        cost = v2v2 + gamma*(v1v2 - v2v2)
        return (gamma, cost)

    def _min_norm_2d(vecs, dps):
        """
        Find the minimum norm solution as combination of two points
        This is correct only in 2D
        ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
        """
        dmin = 1e8
        for i in range(len(vecs)):
            for j in range(i+1,len(vecs)):
                if (i,j) not in dps:
                    dps[(i, j)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i,j)] += torch.mul(vecs[i][k], vecs[j][k]).sum().data.cpu()
                    dps[(j, i)] = dps[(i, j)]
                if (i,i) not in dps:
                    dps[(i, i)] = 0.0
                    for k in range(len(vecs[i])):
                        dps[(i,i)] += torch.mul(vecs[i][k], vecs[i][k]).sum().data.cpu()
                if (j,j) not in dps:
                    dps[(j, j)] = 0.0   
                    for k in range(len(vecs[i])):
                        dps[(j, j)] += torch.mul(vecs[j][k], vecs[j][k]).sum().data.cpu()
                c,d = MinNormSolver._min_norm_element_from2(dps[(i,i)], dps[(i,j)], dps[(j,j)])
                if d < dmin:
                    dmin = d
                    sol = [(i,j),c,d]
        return sol, dps

    def _projection2simplex(y):
        """
        Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
        """
        m = len(y)
        sorted_y, _ = torch.sort(y, descending=True)
        tmpsum = 0.0
        tmax_f = (torch.sum(y) - 1.0)/m
        for i in range(m-1):
            tmpsum+= sorted_y[i]
            tmax = (tmpsum - 1) / (i+1.0)
            if tmax > sorted_y[i + 1]:
                tmax_f = tmax
                break
        return torch.maximum(y - tmax_f, torch.zeros(y.shape))
    
    def _next_point(cur_val, grad, n):
        proj_grad = grad - ( torch.sum(grad) / n )
        tm1 = -1.0*cur_val[proj_grad<0]/proj_grad[proj_grad<0]
        tm2 = (1.0 - cur_val[proj_grad>0])/(proj_grad[proj_grad>0])
        
        skippers = torch.sum(tm1<1e-7) + torch.sum(tm2<1e-7)
        t = 1
        if len(tm1[tm1>1e-7]) > 0:
            t = torch.min(tm1[tm1>1e-7])
        if len(tm2[tm2>1e-7]) > 0:
            t = min(t, torch.min(tm2[tm2>1e-7]))

        next_point = proj_grad*t + cur_val
        next_point = MinNormSolver._projection2simplex(next_point)
        return next_point

    def find_min_norm_element(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)
        
        n= len(vecs)  # num of tasks
        sol_vec = torch.zeros(n)  # weights of tasks
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec , init_sol[2]
    
        iter_count = 0

        grad_mat = torch.zeros((n,n))
        for i in range(n):
            for j in range(n):
                grad_mat[i,j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            grad_dir = -1.0 * torch.matmul(grad_mat, sol_vec)
            new_point = MinNormSolver._next_point(sol_vec, grad_dir, n)
            # Re-compute the inner products for line search
            v1v1 = 0.0
            v1v2 = 0.0
            v2v2 = 0.0
            for i in range(n):
                for j in range(n):
                    v1v1 += sol_vec[i]*sol_vec[j]*dps[(i,j)]
                    v1v2 += sol_vec[i]*new_point[j]*dps[(i,j)]
                    v2v2 += new_point[i]*new_point[j]*dps[(i,j)]
            nc, nd = \
                MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc*sol_vec + (1-nc)*new_point
            change = new_sol_vec - sol_vec
            if torch.sum(torch.abs(change)).item() < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec
        
        

    def find_min_norm_element_FW(vecs):
        """
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric, and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j; the solution lies in (0, d_{i,j})
        Hence, we find the best 2-task solution, and then run the Frank Wolfe until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolver._min_norm_2d(vecs, dps)

        n=len(vecs)
        sol_vec = torch.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec , init_sol[2]

        iter_count = 0

        grad_mat = torch.zeros((n,n))
        for i in range(n):
            for j in range(n):
                grad_mat[i,j] = dps[(i, j)]

        while iter_count < MinNormSolver.MAX_ITER:
            t_iter = torch.argmin(torch.dot(grad_mat, sol_vec))

            v1v1 = torch.dot(sol_vec, torch.dot(grad_mat, sol_vec))
            v1v2 = torch.dot(sol_vec, grad_mat[:, t_iter])
            v2v2 = grad_mat[t_iter, t_iter]

            nc, nd = MinNormSolver._min_norm_element_from2(v1v1, v1v2, v2v2)
            new_sol_vec = nc*sol_vec
            new_sol_vec[t_iter] += 1 - nc

            change = new_sol_vec - sol_vec
            if torch.sum(torch.abs(change)) < MinNormSolver.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec


def gradient_normalizers(grads, losses, normalization_type):
    gn = {}
    if normalization_type == 'l2':
        for t in grads:
            gn[t] = torch.sqrt(torch.sum([gr.pow(2).sum().data.cpu().numpy() for gr in grads[t]]))
    elif normalization_type == 'loss':
        for t in grads:
            gn[t] = losses[t]
    elif normalization_type == 'loss+':
        for t in grads:
            gn[t] = losses[t] * np.sqrt(np.sum([gr.pow(2).sum().data.cpu().numpy() for gr in grads[t]]))
    elif normalization_type == 'none':
        for t in grads:
            gn[t] = 1.0
    else:
        print('ERROR: Invalid Normalization Type')
    return gn



if __name__ == "__main__":
    # 假设我们有3个资产，它们的历史收益向量（单位时间内的回报率）
    returns_asset1 = torch.tensor([0.02, 0.01, -0.01, 0.03, 0.02])  # 资产1的历史收益
    returns_asset2 = torch.tensor([-0.01, 0.03, 0.01, -0.02, 0.01]) # 资产2的历史收益
    returns_asset3 = torch.tensor([0.01, 0.02, 0.01, 0.00, -0.01])  # 资产3的历史收益

    # 计算协方差矩阵（用作风险度量的一部分）
    cov_matrix = torch.cov(torch.stack([returns_asset1, returns_asset2, returns_asset3]))

    # 构造每个资产的风险向量（即每个资产的协方差向量）
    risk_asset1 = cov_matrix[0]  # 资产1的协方差
    risk_asset2 = cov_matrix[1]  # 资产2的协方差
    risk_asset3 = cov_matrix[2]  # 资产3的协方差
    
    print("风险向量:")
    print(risk_asset1)
    print(risk_asset2)
    print(risk_asset3)
    

    # 将这些风险向量作为输入，使用 MinNormSolver 来找到最优的投资组合权重，使组合风险最小
    risks = [risk_asset1, risk_asset2, risk_asset3]

    # 使用 MinNormSolver 来找到最小风险的投资组合权重
    weights, min_risk = MinNormSolver.find_min_norm_element(risks)

    asset = weights[0] * risk_asset1 + weights[1] * risk_asset2 + weights[2] * risk_asset3
    print("2 NORM : ", (asset * asset).sum())

    print("最优投资组合权重:", weights)
    print("最小风险值:", min_risk)