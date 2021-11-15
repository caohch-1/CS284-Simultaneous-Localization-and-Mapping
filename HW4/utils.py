import numpy as np
from scipy.sparse import lil_matrix


def rotate(points, rot_vecs):
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


def project(points, camera_params):
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj ** 2, axis=1)
    r = 1 + k1 * n + k2 * n ** 2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj


def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    return (points_proj - points_2d).ravel()


def least_squares(
        fun, x0, jac='2-point', bounds=(-np.inf, np.inf), kwargs={}):
    loss_function = construct_loss_function(m, loss, f_scale)
    if callable(loss):
        rho = loss_function(f0)
        if rho.shape != (3, m):
            raise ValueError("The return value of `loss` callable has wrong "
                             "shape.")
        initial_cost = 0.5 * np.sum(rho[0])
    elif loss_function is not None:
        initial_cost = loss_function(f0, cost_only=True)
    else:
        initial_cost = 0.5 * np.dot(f0, f0)

    if callable(jac):
        J0 = jac(x0, *args, **kwargs)

        if issparse(J0):
            J0 = csr_matrix(J0)

            def jac_wrapped(x, _=None):
                return csr_matrix(jac(x, *args, **kwargs))

        elif isinstance(J0, LinearOperator):
            def jac_wrapped(x, _=None):
                return jac(x, *args, **kwargs)

        else:
            J0 = np.atleast_2d(J0)

            def jac_wrapped(x, _=None):
                return np.atleast_2d(jac(x, *args, **kwargs))

    else:  # Estimate Jacobian by finite differences.

        J0 = jac_wrapped = None

        jac_sparsity = check_jac_sparsity(jac_sparsity, m, n)

        def jac_wrapped(x, f):
            J = approx_derivative(fun, x, rel_step=diff_step, method=jac,
                                  f0=f, bounds=bounds, args=args,
                                  kwargs=kwargs, sparsity=jac_sparsity)
            if J.ndim != 2:  # J is guaranteed not sparse.
                J = np.atleast_2d(J)

            return J

        J0 = jac_wrapped(x0, f0)

    if J0 is not None:
        jac_scale = isinstance(x_scale, str) and x_scale == 'jac'

        if tr_solver is None:
            if isinstance(J0, np.ndarray):
                tr_solver = 'exact'
            else:
                tr_solver = 'lsmr'

        result = trf(fun_wrapped, jac_wrapped, x0)

    result.message = TERMINATION_MESSAGES[result.status]
    result.success = result.status > 0

    return result


def tr(fun, jac, x0, f0, J0):
    x = x0.copy()

    f = f0
    f_true = f.copy()
    nfev = 1

    J = J0
    njev = 1
    m, n = J.shape

    if loss_function is not None:
        rho = loss_function(f)
        cost = 0.5 * np.sum(rho[0])
    else:
        cost = 0.5 * np.dot(f, f)

    g = compute_grad(J, f)

    jac_scale = isinstance(x_scale, str) and x_scale == 'jac'
    if jac_scale:
        scale, scale_inv = compute_jac_scale(J)
    else:
        scale, scale_inv = x_scale, 1 / x_scale

    v, dv = CL_scaling_vector(x, g, lb, ub)
    v[dv != 0] *= scale_inv[dv != 0]
    Delta = norm(x0 * scale_inv / v ** 0.5)
    if Delta == 0:
        Delta = 1.0

    g_norm = norm(g * v, ord=np.inf)

    f_augmented = np.zeros((m + n))

    alpha = 0.0

    termination_status = None
    iteration = 0
    step_norm = None
    actual_reduction = None

    while True:
        v, dv = CL_scaling_vector(x, g, lb, ub)

        g_norm = norm(g * v, ord=np.inf)
        if g_norm < gtol:
            termination_status = 1

        if termination_status is not None or nfev == max_nfev:
            break

        v[dv != 0] *= scale_inv[dv != 0]

        d = v ** 0.5 * scale

        diag_h = g * dv * scale

        g_h = d * g

        f_augmented[:m] = f
        if tr_solver == 'exact':
            J_augmented[:m] = J * d
            J_h = J_augmented[:m]
            J_augmented[m:] = np.diag(diag_h ** 0.5)
            U, s, V = svd(J_augmented, full_matrices=False)
            V = V.T
            uf = U.T.dot(f_augmented)
        elif tr_solver == 'lsmr':
            J_h = right_multiplied_operator(J, d)

            if regularize:
                a, b = build_quadratic_1d(J_h, g_h, -g_h, diag=diag_h)
                to_tr = Delta / norm(g_h)
                ag_value = minimize_quadratic_1d(a, b, 0, to_tr)[1]
                reg_term = -ag_value / Delta ** 2

            lsmr_op = regularized_lsq_operator(J_h, (diag_h + reg_term) ** 0.5)
            gn_h = lsmr(lsmr_op, f_augmented, **tr_options)[0]
            S = np.vstack((g_h, gn_h)).T
            JS = J_h.dot(S)  # LinearOperator does dot too.
            B_S = np.dot(JS.T, JS) + np.dot(S.T * diag_h, S)
            g_S = S.T.dot(g_h)

        theta = max(0.995, 1 - g_norm)

        actual_reduction = -1
        while actual_reduction <= 0 and nfev < max_nfev:
            p = d * p_h  # Trust-region solution in the original space.
            step, step_h, predicted_reduction = select_step(
                x, J_h, diag_h, g_h, p, p_h, d, Delta, lb, ub, theta)

            x_new = make_strictly_feasible(x + step, lb, ub, rstep=0)
            f_new = fun(x_new)
            nfev += 1

            step_h_norm = norm(step_h)

            if not np.all(np.isfinite(f_new)):
                Delta = 0.25 * step_h_norm
                continue

            if loss_function is not None:
                cost_new = loss_function(f_new, cost_only=True)
            else:
                cost_new = 0.5 * np.dot(f_new, f_new)
            actual_reduction = cost - cost_new
            Delta_new, ratio = update_tr_radius(
                Delta, actual_reduction, predicted_reduction,
                step_h_norm, step_h_norm > 0.95 * Delta)

            step_norm = norm(step)
            termination_status = check_termination(
                actual_reduction, cost, step_norm, norm(x), ratio, ftol, xtol)
            if termination_status is not None:
                break

            alpha *= Delta / Delta_new
            Delta = Delta_new

        if actual_reduction > 0:
            x = x_new

            f = f_new
            f_true = f.copy()

            cost = cost_new

            J = jac(x, f)
            njev += 1

            if loss_function is not None:
                rho = loss_function(f)
                J, f = scale_for_robust_loss_function(J, f, rho)

            g = compute_grad(J, f)

            if jac_scale:
                scale, scale_inv = compute_jac_scale(J, scale_inv)
        else:
            step_norm = 0
            actual_reduction = 0

        iteration += 1

    if termination_status is None:
        termination_status = 0

    active_mask = find_active_constraints(x, lb, ub, rtol=xtol)
    return OptimizeResult(
        x=x, cost=cost, fun=f_true, jac=J, grad=g, optimality=g_norm,
        active_mask=active_mask, nfev=nfev, njev=njev,
        status=termination_status)


def ba_matrix(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

    return A
