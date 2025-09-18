import numpy as np


def lmo_nuclear_norm_ball():
    """
    The Frank-Wolfe lmo for matrix completion problem.
    """

    def f(g):
        U, S, Vh = np.linalg.svd(g, full_matrices=False)
        return np.outer(U[:, 0], Vh[0])

    return lambda g: f(g)


def lmo_l2_ball(radius, center=None):
    r"""
    The Frank-Wolfe lmo function for the l2 ball:
    x \in {x : ||x - center||_2 <= radius}
    
    Parameters:
    -----------
    radius : float
        Radius of the l2 ball
    center : array-like or None
        Center of the ball. If None, center is set to zero vector
        
    Returns:
    --------
    function
        Linear minimization oracle that projects onto the l2 ball
    """

    def f(g):
        if center is None:
            center_p = np.zeros_like(g)
        else:
            center_p = np.broadcast_to(center, g.shape)
            
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-10:
            return center_p
            
        s = center_p - radius * g/g_norm
        
        assert abs(np.linalg.norm(s - center_p) - radius) <= 1e-10, \
            "Solution does not lie on ball boundary"
            
        return s

    return lambda g: f(g)


def lmo_l2_ball_positive_orthant(radius, center=None, epsilon=0.0):
    """
    Frank-Wolfe LMO for an L2 ball in the positive orthant, possibly strictly positive.
    
    Solves:
        min_x <g, x> s.t. x_i >= epsilon and ||x - center||_2 <= radius
    
    Parameters:
    -----------
    radius : float
        Radius of the L2 ball
    center : array-like or None
        Center of the ball. If None, assumed to be the zero vector
    epsilon : float
        If > 0, the solution x must satisfy x_i >= epsilon (strict nonnegativity)
        If == 0, enforces standard x_i >= 0
    
    Returns:
    --------
    function
        Linear minimization oracle g -> s
    """
    def f(g):
        g = np.asarray(g)
        center_p = np.zeros_like(g) if center is None else np.asarray(center)
        assert center_p.shape == g.shape, "Shape mismatch between g and center"

        # Mask for coordinates where decreasing g is beneficial (c_i < 0)
        mask = g < 0
        if not np.any(mask):
            return np.maximum(center_p, epsilon)

        g_neg = g[mask]
        direction = np.zeros_like(g)
        direction[mask] = -g_neg / np.linalg.norm(g_neg)

        # Step from center
        s = center_p + radius * direction

        # Project onto feasible set: x_i >= epsilon
        s = np.maximum(s, epsilon)

        # Safety check
        assert np.all(s >= epsilon), "Output violates epsilon-nonnegativity"
        assert np.linalg.norm(s - center_p) <= radius + 1e-8, "Output outside L2 ball"

        return s

    return f



def lmo_linf_ball(radius, center=None):
    """
    The Frank-Wolfe linear minimization oracle (LMO) for the l_∞ ball:
    x ∈ {x : ||x - center||_∞ <= radius}
    
    Parameters:
    -----------
    radius : float
        Radius of the l_∞ ball
    center : array-like or None
        Center of the ball. If None, center is set to zero vector
        
    Returns:
    --------
    function
        Linear minimization oracle that returns a vertex of the l_∞ ball
    """
    def f(g):
        if center is None:
            center_p = np.zeros_like(g)
        else:
            center_p = np.array(center)
            
        # For l_∞ ball, the solution is a vertex where we move radius units 
        # in the opposite direction of the gradient's sign
        s = center_p - radius * np.sign(g)
        return s

    return lambda g: f(g)


def lmo_simplex(radius=1):
    """
    The Frank-Wolfe Linear Minimization Oracle (LMO) for the simplex constraint.

    Parameters:
    -----------
    radius : float, optional
        The radius of the simplex. Default is 1.

    Returns:
    --------
    function
        A function that takes a gradient vector `g` and returns the vertex of the simplex
        that minimizes the linear approximation of the objective function.
    """

    def f(g):
        s = np.zeros(g.shape)
        s += 1e-15
        min_indices = np.where(g == np.min(g))[0]
        s[min_indices[0]] = radius
        return s

    return f


def lmo_matrix_simplex(radius=1.0):
    """
    Frank-Wolfe LMO for the matrix simplex constraint:
    { X in R^{n x m} | X >= 0, sum(X) = radius }

    Parameters:
    -----------
    radius : float, optional
        The total sum of the elements in the matrix (i.e., the 'radius' of the simplex).

    Returns:
    --------
    function
        A function that takes a gradient matrix `G` and returns a matrix `S` in the
        matrix simplex that minimizes <G, S>.
    """
    
    def f(G):
        S = np.zeros_like(G)
        S += 1e-60
        min_index = np.unravel_index(np.argmin(G), G.shape)
        S[min_index] = radius
        return S

    return f


def lmo_matrix_box(lower, upper):
    """
    Frank-Wolfe LMO for general box constraints on a matrix:
    { X in R^{n x m} | lower[i][j] <= X[i][j] <= upper[i][j] }

    Parameters:
    -----------
    lower : ndarray
        Lower bounds matrix of shape (n, m)
    upper : ndarray
        Upper bounds matrix of shape (n, m)

    Returns:
    --------
    function
        A function that takes a gradient matrix `G` and returns a matrix `S` in the
        box constraint that minimizes <G, S>.
    """
    def f(G):
        S = np.where(G < 0, upper, lower)
        return S

    return f
