from util.helpers import *
from util.sparse import spdiag, spcoo, spblockdiag, extractRowsFromSamplingMat


def samplesFromStrategies(sampling_strategies, num_labels, label_space_dim, random_label_samples, meank):
    samples = 0
    if "exhaustive" in sampling_strategies:
        samples += num_labels ** label_space_dim
    if "random" in sampling_strategies:
        samples += random_label_samples
    if "meank" in sampling_strategies:
        samples += meank + 1  # + 1 for guaranted sample at mean of current solution

    return samples


# Construct label_space_dim interpolation matrices W1i, which compute the pixel value at each sampled label.
# The list is of size label_space_dim, where each element ins a matrix of size=(num_faces * samples, num_faces * num_labels)
def createSamplingMatrices(sampling_strategies, label_space_dim, num_labels, num_faces,
                           random_label_samples, meank,
                           meank_variance, cost_data, interp_qs, u, labels, problem):
    cost_valid = torch.any(cost_data != float("Inf"))

    meank_sampled = meank * ("meank" in sampling_strategies)

    for dd in range(label_space_dim):
        # collect sampling strategies for single q_i in a list
        interp_q = []

        if "random" in sampling_strategies:
            # use more random samples, if cost is not valid (cost is invalid in first iteration)
            samples = random_label_samples
            interp_q.append(randomSampling(num_labels, num_faces, samples))

        if meank_sampled:
            interp_q.append(meankSampling(meank, meank_variance, u, labels, dd, problem))

        if "exhaustive" in sampling_strategies:
            interp_q.append(exhaustiveSampling(label_space_dim, num_labels, num_faces, dd))

        # vstack sampling strategies for single q_i, resulting vstacked matrix corresponds to W_1i
        interp_qs[dd] = torch.vstack(interp_q).coalesce()

    interp_qs_t = [interp_qs[dd].transpose(0, 1).coalesce() for dd in range(label_space_dim)]

    return interp_qs, interp_qs_t


# sample all label points
def exhaustiveSampling(label_space_dim, num_labels, num_faces, dd):
    # for easier row extraction of sparse sampling matrices during sampling process,
    # create sparse matrix with explicit 0, instead of speye(num_faces)
    mat1 = spdiag([torch.zeros(num_faces - 1), torch.ones(num_faces)], [-1, 0],
                  (num_faces, num_faces))
    # manually add 0 in first row, second column
    values_new = torch.cat((mat1.values(), torch.tensor([0])))
    indices_new = torch.cat((mat1.indices(), torch.tensor([[0], [1]])), 1)
    speye2 = spcoo(indices_new[0, :], indices_new[1, :], values_new, mat1.shape)

    inner_reps = num_labels ** (label_space_dim - dd - 1)
    outer_reps = num_labels ** dd
    w_1i = spblockdiag([torch.vstack([speye2] * inner_reps).coalesce()] * num_labels)
    return torch.vstack([w_1i] * outer_reps).coalesce()


def randomSampling(num_labels, num_faces, samples):
    cols_ = torch.randint(num_labels - 1, (num_faces * samples,))
    cols = torch.hstack([cols_, cols_ + 1]) * num_faces
    cols += torch.tensor(range(num_faces)).repeat(2 * samples)
    rows = torch.tensor(range(samples * num_faces)).repeat(2)
    values_ = torch.rand(samples * num_faces)
    values = torch.hstack((values_, 1 - values_))
    return torch.sparse_coo_tensor(indices=torch.vstack((rows, cols)), values=values,
                                   size=(num_faces * samples, num_faces * num_labels),
                                   dtype=torch.float).coalesce()

def getSamplingMatFromUnliftedU(gamma_i, labels_i):
    num_faces = gamma_i.shape[0]
    num_labels = labels_i.shape[0]

    cols_ = torch.zeros(num_faces, dtype=torch.long)
    values_ = torch.zeros(num_faces, dtype=torch.float)
    for ll in range(num_labels - 1):
        ind = torch.logical_and(gamma_i >= labels_i[ll], gamma_i <= labels_i[ll + 1])
        alpha = (gamma_i - labels_i[ll]) / (labels_i[ll + 1] - labels_i[ll])
        cols_[ind] = torch.where(ind)[0] + ll * num_faces
        values_[ind] = alpha[ind]

    rows = torch.tensor(range(num_faces)).repeat(2)
    cols = torch.hstack([cols_, cols_ + num_faces])
    values = torch.hstack((1 - values_, values_))

    return torch.sparse_coo_tensor(indices=torch.vstack((rows, cols)), values=values,
                                   size=(num_faces, num_faces * num_labels),
                                   dtype=torch.float).coalesce()


def meankSampling(meank, meank_variance, u, labels, dd, problem):
    if problem == "hsv" and dd == 0:
        gamma = recoverUManifold(u[:, dd:dd+1], labels[:, dd:dd+1])[:, dd]  # size=(num_faces, label_space_dim)
    else:
        gamma = recoverU(u[:, dd:dd+1], labels[:, dd:dd+1])[:, 0]  # size=(num_faces, label_space_dim)
    num_faces = gamma.shape[0]

    sampling_mats = [getSamplingMatFromUnliftedU(gamma, labels[:, dd])]

    label_length = labels[-1, dd] - labels[0, dd]
    variance = meank_variance / (label_length * label_length)

    gamma_meank = gamma.unsqueeze(0).repeat((meank, 1))

    proposed_samples = gamma_meank + torch.normal(0, torch.sqrt(variance), (meank, num_faces))
    for i in range(1000):
        out_boundary_ind = torch.where((proposed_samples > labels[-1, dd]) | (proposed_samples < labels[0, dd]))
        if out_boundary_ind[0].shape[0] == 0 :
            break
        proposed_samples[out_boundary_ind[0], out_boundary_ind[1]] = \
                    gamma_meank[out_boundary_ind[0], out_boundary_ind[1]] \
                    + torch.normal(0, torch.sqrt(variance), (meank, num_faces))[out_boundary_ind[0], out_boundary_ind[1]]

    samples = proposed_samples

    for n in range(meank):
        sampling_mats.append(getSamplingMatFromUnliftedU(samples[n, :], labels[:, dd]))

    return torch.vstack(sampling_mats)


def getPruningIndices(interp_qs, q, cost_data):
    label_space_dim = len(interp_qs)

    # compute W_1 * q
    interped_qs = sum([interp_qs[ii].mm(q[:, ii].unsqueeze(1)) for ii in range(label_space_dim)])

    # calculate constraints (constraint is inactive if it's positive)
    constraints = cost_data - interped_qs

    # get the index where the constraints are active
    # keep all active constraints, i.e. remove desired inactive constraints
    keep_ind = torch.where(constraints < 0)[0]  # remove all inactive

    return keep_ind


def getViolatedConstraintIndices(interp_qs, q, cost_data, mostviok, num_faces, mostviok_pixel):
    label_space_dim = len(interp_qs)
    # compute W_1 * q
    interped_qs = sum([interp_qs[ii].mm(q[:, ii].unsqueeze(1)) for ii in range(label_space_dim)])

    # calculate constraints (constraint is violated if it's negative)
    constraints = cost_data - interped_qs
    constraints_img = vec2img(constraints, -1, num_faces).squeeze()

    # only consider mostviok most violated constraints
    # get the index where the constraints are violated
    vio_ind = torch.where(constraints < 0.0)[0]
    if mostviok > 0 and vio_ind.numel() > 0:
        if mostviok_pixel:
            mostviok_ = min(constraints_img.shape[-1], mostviok)
            # get smallest constraints per face
            mostvio_ind = \
                torch.topk(constraints_img, mostviok_, dim=-1, largest=False, sorted=False)[1]
            # make linear indices
            mostvio_ind *= num_faces
            mostvio_ind += torch.arange(constraints_img.shape[0]).unsqueeze(1)
            mostvio_ind = img2vec(mostvio_ind)
            # only choose violated constraints
            vio_ind = mostvio_ind[constraints[mostvio_ind].squeeze() < 0.0].reshape(-1)
        else:  # mostviok for all samples
            vio_const = constraints[vio_ind]
            mostviok_ = min(vio_const.shape[0], mostviok)
            mostvio_ind = torch.topk(vio_const, mostviok_, dim=0, largest=False, sorted=False)[1]
            vio_ind = vio_ind[mostvio_ind].reshape(-1)

    return vio_ind


def samplingAcceptance(cost_data, interp_qs, cost_data_prev, interp_qs_prev, q, v1, mostviok,
                       mostviok_pixel, pruning, one_sample_mean, 
                       constraints, num_faces, u, labels, cost, img_vec, problem):
    label_space_dim = len(interp_qs)
    num_labels = labels.shape[0]

    if mostviok==0:
        vio_ind = torch.arange(0, cost_data.shape[0])
    else:
        vio_ind = getViolatedConstraintIndices(interp_qs, q, cost_data, mostviok, num_faces, mostviok_pixel)
    if one_sample_mean:
        samples_u = []
        if problem == "hsv":
            samples_u.append(getSamplingMatFromUnliftedU(recoverUManifold(u[:, 0:1], labels[:, 0:1])[:, 0], labels[:, 0]))
            for i in range(1, label_space_dim):
                samples_u.append(getSamplingMatFromUnliftedU(recoverU(u[:, i:i+1], labels[:, i:i+1])[:, 0], labels[:, i]))
        else:
            for i in range(label_space_dim):
                samples_u.append(getSamplingMatFromUnliftedU(recoverU(u, labels)[:, i], labels[:, i]))
        samples_cost = cost(liftInputImgs(img_vec, 1), recoverGamma(samples_u, labels))
        if problem == "hsv":
            row_indices = samples_u[0].indices()[0]
            col_indices = samples_u[0].indices()[1]
            val = samples_u[0].values()
            col_indices[col_indices >= num_faces * (num_labels - 1)] -= num_faces * (num_labels - 1)
            samples_u[0] = spcoo(row_indices, col_indices, val, samples_u[0].shape)
        new_constraints = vio_ind.shape[0] + num_faces
    else:
        samples_cost = torch.tensor([[]]).transpose(0, 1)
        new_constraints = vio_ind.shape[0]

    if constraints.numel():
        constraints.put_(vio_ind % constraints.numel(), torch.ones(vio_ind.shape), accumulate=True)

    if pruning:
        keep_ind = getPruningIndices(interp_qs_prev, q, cost_data_prev)
        removed_constraints = cost_data_prev.shape[0] - keep_ind.shape[0]
        cost_data = torch.vstack([cost_data_prev[keep_ind], cost_data[vio_ind], samples_cost])
        v1 = torch.vstack([v1[keep_ind], torch.zeros((new_constraints, 1))])
        for i in range(label_space_dim):
            if one_sample_mean:
                interp_qs[i] = torch.vstack([extractRowsFromSamplingMat(interp_qs_prev[i], keep_ind),
                                             extractRowsFromSamplingMat(interp_qs[i],
                                                                        vio_ind),
                                             samples_u[i]]).coalesce()
            else:
                interp_qs[i] = torch.vstack([extractRowsFromSamplingMat(interp_qs_prev[i], keep_ind),
                                             extractRowsFromSamplingMat(interp_qs[i],
                                                                        vio_ind)]).coalesce()
    else:  # cheaper to not slice data
        removed_constraints = 0
        cost_data = torch.vstack([cost_data_prev, cost_data[vio_ind]])
        v1 = torch.vstack([v1, torch.zeros((new_constraints, 1))])
        for i in range(label_space_dim):
            interp_qs[i] = torch.vstack([interp_qs_prev[i],
                                         extractRowsFromSamplingMat(interp_qs[i],
                                                                    vio_ind)]).coalesce()

    interp_qs_t = [interp_qs[dd].transpose(0, 1).coalesce() for dd in range(label_space_dim)]

    return new_constraints, removed_constraints, cost_data, v1, interp_qs, interp_qs_t, constraints


def violationSampling(outer_iter, sampling, num_labels, random_label_samples,
                      meank, cost_data, meank_variance, interp_qs, u, labels, img_vec, width,
                      height, q, v1, cost, mostviok, mostviok_pixel, pruning,
                      one_sample_mean, constraints, problem):
    label_space_dim = len(interp_qs)

    if outer_iter == 0:
        sampling_strategy = ['exhaustive']
        interp_qs_prev = [None] * label_space_dim
    else:
        # save old sampling matrix (only needed if outer_iter > 0, i.e. when interp_qs_prev contains valid values)
        interp_qs_prev = [interp_qs[dd].detach().clone() for dd in range(label_space_dim)]
        sampling_strategy = sampling

    # construct proposal sampling matrix (random and meank only)
    num_faces = numFaces(height, width)
    interp_qs, interp_qs_t = createSamplingMatrices(sampling_strategy, label_space_dim,
                                                    num_labels, num_faces,
                                                    random_label_samples, meank,
                                                    meank_variance, cost_data,
                                                    interp_qs, u, labels, problem)

    # save old cost (only needed if outer_iter > 0, i.e. when cost_data_prev contains valid values)
    cost_data_prev = cost_data.detach().clone()

    # compute cost of data term using proposal sampling matrix
    proposed_samples = int(interp_qs[0].shape[0] / num_faces)
    cost_data = cost(liftInputImgs(img_vec, proposed_samples), recoverGamma(interp_qs, labels))

    if problem == "hsv":
        row_indices = interp_qs[0].indices()[0]
        col_indices = interp_qs[0].indices()[1]
        val = interp_qs[0].values()
        col_indices[col_indices >= num_faces * (num_labels - 1)] -= num_faces * (num_labels - 1)
        interp_qs[0] = spcoo(row_indices, col_indices, val, interp_qs[0].shape)
        interp_qs_t[0] = interp_qs[0].transpose(0, 1).coalesce()

    if outer_iter == 0:
        new_constraints = interp_qs[0].shape[0]
        # initialize the new lagrange multipliers
        v1 = torch.zeros((new_constraints, 1), dtype=torch.float)
        removed_constraints = 0
        if constraints.numel():
            constraints[:] += proposed_samples
    else:
        # exhaustive_num is hardcoded 0 for now, as we did not yet implement pruning
        new_constraints, removed_constraints, cost_data, v1, interp_qs, interp_qs_t, constraints = samplingAcceptance(
            cost_data, interp_qs, cost_data_prev, interp_qs_prev, q, v1, mostviok, mostviok_pixel,
            pruning, one_sample_mean, constraints, num_faces, u, labels, cost, img_vec, problem)

    current_constraints = v1.shape[0]
    return current_constraints, new_constraints, removed_constraints, cost_data, v1, interp_qs, interp_qs_t, constraints

