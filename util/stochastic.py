import time

from util.helpers import *
from util.optimization import *
from util.sampling import samplesFromStrategies, createSamplingMatrices, violationSampling
from util.sparse import *
import matplotlib.pyplot as plt


def preconditioner(diff, interp_qs_t, interp_p_t, interp_p_manifold_t, h_x2):
    label_space_dim = len(interp_qs_t)

    diff_abs = spabs(diff)
    interp_qs_t_abs = spabs(torch.vstack(interp_qs_t).coalesce())
    interp_p_t_abs = spabs(interp_p_t)
    interp_p_manifold_t_abs = spabs(interp_p_manifold_t)

    dimsum(diff_abs, 0)
    diff_rowsum = dimsum(diff_abs, 0)
    diff_colsum = dimsum(diff_abs, 1)
    interp_p_t_rowsum = dimsum(interp_p_t_abs, 0)
    interp_p_t_colsum = dimsum(interp_p_t_abs, 1)
    interp_qs_t_rowsum = dimsum(interp_qs_t_abs, 0)
    interp_qs_t_colsum = dimsum(interp_qs_t_abs, 1)
    interp_qs_t_colsum = vec2img(interp_qs_t_colsum, label_space_dim, -1).squeeze(0)
    interp_p_manifold_t_rowsum = torch.sparse.sum(interp_p_manifold_t_abs, dim=0).to_dense().reshape(-1, 1)
    interp_p_manifold_t_colsum = torch.sparse.sum(interp_p_manifold_t_abs, dim=1).to_dense().reshape(-1, 1)

    u_precond = 1.0 / (h_x2 * (1 + diff_rowsum))  # OK
    v1_precond = 1.0 / interp_qs_t_rowsum  # OK
    v2_precond = 1.0 / interp_p_t_rowsum  # OK
    v2_manifold_precond = 1.0 / (interp_p_manifold_t_rowsum)
    v2_manifold_precond[torch.isinf(v2_manifold_precond)] = 1e8
    q_precond = 1.0 / (h_x2 + interp_qs_t_colsum)  # OK
    p_precond = 1.0 / (interp_p_t_colsum + h_x2 * diff_colsum)  # OK
    p_manifold_precond = 1.0 / (interp_p_manifold_t_colsum + h_x2 * diff_colsum)

    return u_precond, v1_precond, v2_precond, q_precond, p_precond, v2_manifold_precond, p_manifold_precond

def getStepSizes(diff, interp_qs_t, interp_p_t, interp_p_manifold_t, h_x2):
    tau_s = 0.8
    tau_t = 0.8
    theta = 1
    (u_precond, v1_precond, v2_precond, q_precond, 
     p_precond, v2_manifold_precond, p_manifold_precond) = preconditioner(diff, interp_qs_t,
                                                                          interp_p_t, interp_p_manifold_t, h_x2)

    return theta, tau_s, tau_t, u_precond, v1_precond, v2_precond, q_precond, p_precond, v2_manifold_precond, p_manifold_precond


def updatePrimalU(u, tau_s, dir, u_prev_outer, num_labels, num_faces, problem):
    u = gradientDescentStep(u, tau_s, dir)
    return proxIndicatorSimplex(u, num_labels, num_faces, problem)

def updateLagrangeV1(v1, tau_s, dir):
    v1 = gradientDescentStep(v1, tau_s, dir)
    return proxIndicatorPositive(v1)


def updateLagrangeV2(v2, tau_s, dir, vertex_samples, lambda_, variable):
    v2 = gradientDescentStep(v2, tau_s, dir)
    return proxL2(v2, vertex_samples, lambda_, tau_s, variable)


def updateDualVar(dual_var, tau_t, dir, p_prev):
    dual_var = gradientAscentStep(dual_var, tau_t, dir)
    return dual_var


def primalEnergy(cost, u, img_vec, lambda_, h_x, max_iter, verbose=True):
    # energy from regularization term
    energy_reg_per_pixel = primalRegEnergyFixedU(u, lambda_, 10 * max_iter, 1e-6, verbose)
    energy_reg = energy_reg_per_pixel.sum()

    # energy from data term
    energy_data_per_pixel = cost(img_vec, img2vec(u))
    energy_data = energy_data_per_pixel.sum()
    energy = h_x * h_x * (energy_data + lambda_ * energy_reg)
    return energy.item()


def setLoss(key, params):
    losses = {'L1': lambda r: L1loss(r).sum(dim=-1, keepdim=True),
              'trunc_L1': lambda r: truncate(L1loss(r).sum(dim=-1, keepdim=True), params[0]),
              'sqL2': lambda r: sqL2loss(r).sum(dim=-1, keepdim=True),
              'trunc_sqL2': lambda r: truncate(sqL2loss(r), params[0]).sum(dim=-1, keepdim=True)}
    return losses[key]

def stochastic(img_vec, height, width, labels, residual, lambda_, max_iter_inner, max_iter_outer,
               random_label_samples, sampling, meank, meank_variance, mostviok, mostviok_pixel, 
               one_sample_mean, pruning, loss_key, loss_params, problem):
    print("Solve lifted problem with sampling and primal-dual:")
    num_labels, label_space_dim = labels.shape
    num_faces = numFaces(height, width)
    num_edges = numEdges(height, width)
    vertex_samples = vertexSamples(num_faces, num_labels)

    loss = setLoss(loss_key, loss_params)
    cost = lambda f, g: loss(residual(f, g, width, height))

    # discretization scaling variables
    h_x = geth_x(width, height)
    h_x2 = 1.0

    print("Initialize variables")
    # initial primal variables
    u = torch.zeros((num_faces * num_labels, label_space_dim), dtype=torch.float) + 1.0 / num_labels

    # initial dual variables
    q = torch.zeros((num_faces * num_labels, label_space_dim), dtype=torch.float)
    p = torch.zeros((num_edges * num_labels, label_space_dim), dtype=torch.float)
    q_bar = torch.zeros((num_faces * num_labels, label_space_dim), dtype=torch.float)
    p_bar = torch.zeros((num_edges * num_labels, label_space_dim), dtype=torch.float)

    y_u_prev_outer = u.detach().clone()
    y_p_prev_outer = p.detach().clone()
    y_q_prev_outer = q.detach().clone()

    # initialize (empty) lagrange multiplier to allow stacking in case of violation sampling
    v1 = torch.empty((0, 1), dtype=torch.float)
    v2 = torch.zeros((2 * vertex_samples, label_space_dim), dtype=torch.float)

    print("Create linear operators")
    # constant linear operators
    nabla, div = nabla3d(nabla2d(height, width)[0], num_labels)
    interp_p, interp_p_t, interp_p_manifold, interp_p_manifold_t = nablaGamma(interpFEM(height, width)[0], labels, problem, num_edges)

    interp_qs = [None] * label_space_dim
    if "violation" not in sampling:
        samples = samplesFromStrategies(sampling, num_labels, label_space_dim, random_label_samples, meank)
    else:
        samples = num_labels ** label_space_dim

    # initialize cost
    cost_data = float("Inf") + torch.zeros((num_faces * samples, 1), dtype=torch.float)

    constraints = torch.empty(0)

    memory = []

    total_constraints = 0
    # sampling loop
    print("Start optimization")
    print(
        "  {:>10}  {:>10}  {:>11}  {:>19}  {:>15}  {:>19}  {:>17}".format('Outer iter',
                                                                                  'Inner iter',
                                                                                  'Memory [Mb]',
                                                                                  'Current constraints',
                                                                                  'New constraints',
                                                                                  'Removed constraints',
                                                                                  'Total constraints',
                                                                                  ))

    torch.cuda.empty_cache()
    start_outer = time.time()
    for jj in range(max_iter_outer):
        if "violation" in sampling:
            current_constraints, new_constraints, removed_constraints, cost_data, v1, interp_qs, interp_qs_t, constraints = violationSampling(
                jj, sampling, num_labels, random_label_samples, meank, cost_data, meank_variance,
                interp_qs, u, labels, img_vec, width, height, q, v1, cost, mostviok, mostviok_pixel,
                pruning, one_sample_mean, constraints, problem)
            total_constraints += new_constraints

            if new_constraints == 0:  # if no new constraints are added then we have converged
                print(" CONVERGED: No constraints added for next iteration")
                break
        else:
            interp_qs, interp_qs_t = createSamplingMatrices(sampling, label_space_dim, num_labels,
                                                            num_faces, random_label_samples,
                                                            meank, meank_variance,
                                                            cost_data, interp_qs,
                                                            u, labels)
            cost_data = cost(liftInputImgs(img_vec, samples), recoverGamma(interp_qs, labels))
            if problem == 'hsv':
                row_indices = interp_qs[0].indices()[0]
                col_indices = interp_qs[0].indices()[1]
                val = interp_qs[0].values()
                col_indices[col_indices >= num_faces * (num_labels - 1)] -= num_faces * (num_labels - 1)
                interp_qs[0] = spcoo(row_indices, col_indices, val, interp_qs[0].shape)
                interp_qs_t[0] = interp_qs[0].transpose(0, 1).coalesce()

            current_constraints = samples * num_faces

            if constraints.numel():
                constraints[:] = samples
            new_constraints = 0
            removed_constraints = 0
            total_constraints = current_constraints

            # reinitialize the lagrange multipliers
            v1 = torch.zeros((current_constraints, 1), dtype=torch.float)

        # update step size
        (theta, tau_s, tau_t, u_precond, v1_precond, v2_precond, 
         q_precond, p_precond, v2_manifold_precond, p_manifold_precond) = getStepSizes(
            nabla, interp_qs_t, interp_p_t, interp_p_manifold_t, h_x2)

        # proximal loop
        torch.cuda.empty_cache()
        memory.append(torch.cuda.memory_allocated())

        for ii in range(max_iter_inner):
            outer_iter = str(jj) + '/' + str(max_iter_outer - 1)
            inner_iter = str(ii) + '/' + str(max_iter_inner - 1)
            print("\r  {:>10}  {:>10}  {:>11.2f}  {:>19}  {:>15}  {:>19}  {:>17}".format(
                outer_iter,
                inner_iter,
                memory[
                    -1] * 1e-6,
                current_constraints,
                new_constraints,
                removed_constraints,
                total_constraints,
                ),
                end='', flush=True)

            # save old iterates
            u_prev_inner = u.detach().clone()
            v1_prev_inner = v1.detach().clone()
            v2_prev_inner = v2.detach().clone()
            q_prev_inner = q.detach().clone()
            p_prev_inner = p.detach().clone()

            # update primal variables
            interped_qs = sum([interp_qs[i].mm(q_bar[:, i:i + 1]) for i in range(label_space_dim)])
            u = updatePrimalU(u, tau_s * u_precond, h_x2 * (q_bar - div.mm(p_bar)), y_u_prev_outer,
                              num_labels, num_faces, problem)
       
            v1 = updateLagrangeV1(v1, tau_s * v1_precond, cost_data - interped_qs)
            if problem == "hsv":
                v2[:, 0:1] = updateLagrangeV2(v2[:, 0:1], tau_s * v2_manifold_precond, 
                              interp_p_manifold.mm(p_bar[:, 0].unsqueeze(1)), vertex_samples, lambda_, "manifold")
                v2[:, 1:] = updateLagrangeV2(v2[:, 1:], tau_s * v2_precond, interp_p.mm(p_bar[:, 1:]), vertex_samples,
                                      lambda_, None)
            else:
                v2 = updateLagrangeV2(v2, tau_s * v2_precond, interp_p.mm(p_bar), vertex_samples,
                                   lambda_, None)
            # update dual variables
            v1s = -torch.hstack([interp_qs_t[i].mm(v1) for i in range(label_space_dim)])
            q = updateDualVar(q, tau_t * q_precond, h_x2 * u + v1s, y_q_prev_outer)
            if problem != "hsv":
                p = updateDualVar(p, tau_t * p_precond, h_x2 * nabla.mm(u) + interp_p_t.mm(v2),
                                  y_p_prev_outer)
            else:
                nabla_u = h_x2 * nabla.mm(u)
                p[:, 0] = updateDualVar(p[:, 0].unsqueeze(1), tau_t * p_manifold_precond,
                                        nabla_u[:, 0].unsqueeze(1) + interp_p_manifold_t.mm(v2[:, 0:1]),
                                        y_p_prev_outer[:, 0].unsqueeze(1)).squeeze()
                p[:, 1:] = updateDualVar(p[:, 1:], tau_t * p_precond, nabla_u[:, 1:] + interp_p_t.mm(v2[:, 1:]),
                                        y_p_prev_outer[:, 1:])

            # extrapolate dual variables
            q_bar = extrapolation(q, q_prev_inner, theta)
            p_bar = extrapolation(p, p_prev_inner, theta)

        y_u_prev_outer = u.detach().clone()
        y_p_prev_outer = p.detach().clone()
        y_q_prev_outer = q.detach().clone()

    end_outer = time.time()

    final_u = u.detach().clone()

    if problem == "hsv":
        final_result = torch.zeros(3, height, width)
        final_result[0, :, :] = vec2img(recoverUManifold(final_u[:, 0:1], labels[:, 0:1]), width, height)
        final_result[1:, :, :] = vec2img(recoverU(final_u[:, 1:], labels[:, 1:]), width, height)
        primal_energy = primalEnergy(cost, final_result[0, :, :].unsqueeze(0), img_vec, lambda_, h_x, max_iter_inner * max_iter_outer)
    else:
        final_result = vec2img(recoverU(final_u, labels), width, height)
        primal_energy = primalEnergy(cost, final_result, img_vec, lambda_, h_x,
                                     max_iter_inner * max_iter_outer)



    return final_result, primal_energy, end_outer - start_outer, ii, jj, "{:.2f} Mb".format(
        max(memory) * 1e-6), constraints

