# Python fallback helper module for Frangi filters.
# These are used when Cython cannot compile the high-speed/low-memory module.

#pylint: disable=too-many-arguments,too-many-locals,too-many-statements

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__all__ = ['frangi2', 'frangi3']

with_cython = False

########## Frangi 2D ##########
def frangi2(im, out, sigmas, beta, c, black, return_full):
    """
    Internal function to compute the 2D Frangi filter according to Frangi et al (1998).

    See frangi.frangi2 for more details. A few differences in the signature are that all arguments
    are required and a value of 0.0 is used for c to indicate a dynamic value.

    Written by Jeffrey Bush (NCMIR, 2016)
    Adapted from the MATLAB version by M. Schrijver (2001) and D. Kroon (2009)
    """
    from numpy import empty, zeros, putmask, extract, place, copyto, maximum
    from numpy import add, subtract as sub, exp, arctan2, less, greater

    comp = greater if black else less
    beta = -1/(2*beta*beta)
    c = None if c == 0.0 else (-1/(2*c*c))
    first = True

    # Allocate temporaries
    sh = im.shape
    Dxx, Dxy, Dyy, tmp = empty(sh), empty(sh), empty(sh), empty(sh) # 4 TEMPs
    if return_full: sigs, dirs = zeros(sh), zeros(sh)
    mask = empty(sh, bool) # 1 TEMP
    filtered = zeros(sh) # 1 TEMP

    # Frangi filter for all sigmas
    for sigma in sigmas:
        # Calculate the scaled 2D Hessian
        hessian2(im, sigma, Dxx, Dxy, Dyy, tmp)
        
        # Calculate (abs sorted) eigenvalues and vectors
        if return_full:
            lambda1,lambda2,vx,vy = eig2image(Dxx, Dxy, Dyy) # 3 TEMPs + 4 outputs
            angles = arctan2(vy, vx, out=vx)
            del vx, vy
        else:
            lambda1,lambda2 = eigval2image(Dxx, Dxy, Dyy) # 2 TEMPs + 2 outputs
        
        # Compute similarity measures
        comp(lambda2, 0.0, out=mask) # wherever mask is True we have a non-zero output
        lambda1 = extract(mask, lambda1) # 1 TEMP
        lambda2 = extract(mask, lambda2) # 1 TEMP
        lambda1 *= lambda1
        lambda2 *= lambda2

        Rb2 = lambda1 / lambda2 # 1 TEMP
        Rb2 *= beta
        exp(Rb2, out=Rb2) # exp(-Rb^2/(2*beta^2)); Rb = lambda1 / lambda2

        S2 = add(lambda1, lambda2, lambda1)
        S2 *= (-2.0/S2.max()) if c is None else c
        sub(1, exp(S2, out=S2), out=S2) # 1-exp(-S^2/(2*c^2)); S = sqrt(lambda1^2 + lambda2^2)
        
        Rb2 *= S2
        place(filtered, mask, Rb2)
        
        del Rb2, S2, lambda1, lambda2
        
        # Store pixels which are maximal
        if first: copyto(out, filtered)
        elif return_full:
            greater(filtered, out, out=mask)
            putmask(out, mask, filtered)
        else: maximum(filtered, out, out=out)
        if return_full:
            putmask(sigs, mask, sigma)
            putmask(dirs, mask, angles)
            del angles
        
        first = False

    # Return output
    return (out,sigs,dirs) if return_full else out

def hessian2(im, sigma, Dxx, Dxy, Dyy, tmp):
    """
    Filters an image with the 2nd derivatives of a Gaussian with parameter `sigma`. This
    approximates the 2nd derivatives of the image.
    
    `hessian2(im, sigma, Dxx, Dxy, Dyy, tmp)`

    Inputs:
        im      the input image
        sigma   the sigma of the Gaussian kernel used
        
    Outputs are the 2nd derivatives multiplied by sigma^2:
        Dxx     with respect to x both times
        Dxy     with respect to x then y or y then x (equivalent)
        Dyy     with respect to y both times
        tmp     used for temporary storage
    
    The outputs form the matrix:
        | Dxx  Dxy |
        |          |
        | Dxy  Dyy |
    
    Written by Jeffrey Bush (NCMIR, 2016)
    Adapted from the MATLAB version by D. Kroon (2009)
    """
    # Note: we separate the 2D filters into 1D filters and have changed them to be correlate
    # filters instead of convolution filters. Overall this speeds up the code tremendously (over
    # 10x faster, especially since scipy never tries to separate the filters itself). Also, the
    # results are multipled by sigma^2 instead of doing that in the frangi2 code instead to save
    # computation.
    from numpy import arange, exp, pi, sqrt
    from scipy.ndimage import correlate1d
    sigma = float(sigma)
    sigma2,p = sigma*sigma, round(3*sigma)
    i = arange(p, -p-1.0, -1.0)
    i2 = i*i
    v = exp(-i2/(2*sigma2))
    v *= 1.0/(sqrt(2.0*pi)*sigma2)
    uu,u = v*i, v*(i2-sigma2)
    correlate1d(im, uu, 0, tmp); correlate1d(tmp, uu, 1, Dxy)
    correlate1d(im, u,  0, tmp); correlate1d(tmp, v,  1, Dxx)
    correlate1d(im, v,  0, tmp); correlate1d(tmp, u,  1, Dyy)

def eigval2image(Dxx, Dxy, Dyy):
    """
    Calculate the eigenvalues from the Hessian matrix, sorted by absolute value.
    
    Note: this will overwrite the contents of Dxx, Dxy, and Dyy. If these matrices are needed
    later you must give this method a copy of them.
    
    `eig1,eig2 = eigval2image(Dxx,Dxy,Dyy)`
    
    Inputs:
        Dxx,Dxy,Dyy the outputs from hessian2
    
    Outputs:
        eig1        the eigenvalues that are smallest for the entire image
        eig2        the eigenvalues that are largest for the entire image
    """
    from numpy import sqrt, where, absolute, add, subtract as sub
    
    tmp = Dxx - Dyy; tmp *= tmp
    Dxy *= Dxy; Dxy *= 4.0
    tmp += Dxy
    sqrt(tmp, out=tmp)
    
    # Compute the eigenvalues
    Dxx += Dyy
    mu1 = add(Dxx, tmp, out=Dyy); mu1 *= 0.5 # mu1 = (Dxx + Dyy + sqrt((Dxx-Dyy)^2+4*Dxy^2)) / 2
    mu2 = sub(Dxx, tmp, out=Dxx); mu2 *= 0.5 # mu2 = (Dxx + Dyy - sqrt((Dxx-Dyy)^2+4*Dxy^2)) / 2
    
    # Sort eigenvalues by absolute value abs(eig1) < abs(eig2)
    check = absolute(mu1, out=Dxy) > absolute(mu2, out=tmp)
    return where(check, mu2, mu1), where(check, mu1, mu2)

def eig2image(Dxx, Dxy, Dyy):
    """
    Calculate the eigenvalues and eigenvectors from the Hessian matrix, sorted by absolute
    value.
    
    Note: this will overwrite the contents of Dxx, Dxy, and Dyy. If these matrices are needed
    later you must give this method a copy of them.
    
    `eig1,eig2,vx,vy = eig2image(Dxx,Dxy,Dyy)`
    
    Inputs:
        Dxx,Dxy,Dyy the outputs from hessian2

    Outputs:
        eig1        the eigenvalues that are smallest for the entire image
        eig2        the eigenvalues that are largest for the entire image
        vx          the x values of the eigenvector from eig1
        vy          the x values of the eigenvector from eig2
    """
    from numpy import sqrt, where, greater, absolute, add, subtract as sub, multiply as mul, negative

    Dxx_p_Dyy = Dxx + Dyy
    Dxx_m_Dyy = Dxx - Dyy
    tmp = mul(Dxx_m_Dyy, Dxx_m_Dyy, out=Dxx)
    tmp += mul(4.0, mul(Dxy, Dxy, out=Dyy), out=Dyy)
    tmp = sqrt(tmp, out=tmp)
    
    # Compute and normalize the eigenvectors
    v2x = mul(2.0, Dxy, out=Dxy)             # v2x = 2 * Dxy
    v2y = sub(tmp, Dxx_m_Dyy, out=Dxx_m_Dyy) # v2y = sqrt((Dxx-Dyy)^2+4*Dxy^2)) - Dxx + Dyy
    tmp2 = v2y*v2y
    mag = sqrt(add(mul(v2x, v2x, out=Dyy), tmp2, out=Dyy), out=Dyy) # mag = sqrt(v2x^2 + v2y^2)
    B = mag == 0
    mag[B] = 1.0
    v2x /= mag; v2y /= mag

    # Compute the eigenvalues
    mu1 = add(Dxx_p_Dyy, tmp, out=Dyy); mu1 *= 0.5 # mu1 = (Dxx + Dyy + sqrt((Dxx-Dyy)^2+4*Dxy^2)) / 2
    mu2 = sub(Dxx_p_Dyy, tmp, out=tmp); mu2 *= 0.5 # mu2 = (Dxx + Dyy - sqrt((Dxx-Dyy)^2+4*Dxy^2)) / 2
    
    # Sort eigenvalues by absolute value abs(eig1) < abs(eig2)
    check = greater(absolute(mu1, out=Dxx_p_Dyy), absolute(mu2, out=tmp2), B)

    # The eigenvectors are orthogonal
    vx = where(check, v2y, v2x)
    vy = where(check, v2x, negative(v2y, out=v2y))
    return where(check, mu2, mu1),where(check, mu1, mu2), vx,vy


########## Frangi 3D ##########
def frangi3(im, out, sigmas, alpha, beta, c, black, return_full):
    """
    Internal function to compute the 3D Frangi filter according to Frangi et al (1998).

    See frangi.frangi3 for more details. A few differences in the signature are that all arguments
    are required and a value of 0.0 is used for c to indicate a dynamic value.

    Written by Jeffrey Bush (NCMIR, 2016)
    Adapted from the MATLAB version by D. Kroon (2009)
    """
    from numpy import empty, zeros, putmask, extract, place, copyto, maximum
    from numpy import subtract as sub, multiply, divide, exp, greater, less, not_equal, absolute
    
    comp = greater if black else less
    alpha = -1.0/(2.0*alpha*alpha)
    beta = -1.0/(2.0*beta*beta)
    c = None if c == 0.0 else (-1/(2*c*c))
    first = True
    
    # Allocate temporaries
    sh = im.shape
    Dxx, Dxy, Dxz, Dyy, Dyz, Dzz, tmp = (empty(sh), empty(sh), empty(sh),
        empty(sh), empty(sh), empty(sh), empty(sh)) # 7 TEMPs
    if return_full:
        sigs, vecx, vecy, vecz = zeros(sh), zeros(sh), zeros(sh), zeros(sh)
        vx, vy, vz = Dyy, Dyz, Dzz
    mask, msk_tmp = empty(sh, bool), empty(sh, bool) # 2 TEMPs
    filtered = zeros(sh) # 1 TEMP
    
    # Frangi filter for all sigmas
    for sigma in sigmas:
        # Calculate scaled 3D Hessian
        hessian3(im, sigma, Dxx, Dxy, Dxz, Dyy, Dyz, Dzz, tmp)
    
        # Calculate eigenvalues/eigenvectors
        eig3volume(Dxx, Dxy, Dxz, Dyy, Dyz, Dzz, return_full) # 7/21+ TEMPs
        lambda1, lambda2, lambda3 = Dxx, Dxy, Dxz # aliases / reused memory
        
        # Calculate the mask of wherever we have a non-zero output
        mask  = comp(lambda2, 0.0, out=mask)
        mask |= comp(lambda3, 0.0, out=msk_tmp)
        mask &= not_equal(lambda2, 0.0, out=msk_tmp)
        mask &= not_equal(lambda3, 0.0, out=msk_tmp)

        # Calculate absolute values of eigenvalues
        lambda1 = extract(mask, lambda1) # 1 TEMP
        lambda2 = extract(mask, lambda2) # 1 TEMP
        lambda3 = extract(mask, lambda3) # 1 TEMP
        lam23 = lambda2*lambda3 # 1 TEMP
        absolute(lam23, out=lam23)
        lambda1 *= lambda1
        lambda2 *= lambda2
        lambda3 *= lambda3

        # Second order structureness
        S2 = lambda1 + lambda2; S2 += lambda3    # S = sqrt(sum(lambda_i^2)) # 1 TEMP

        # The vesselness features
        Ra2 = divide(lambda2, lambda3, lambda2)  # Ra = |lambda2|/|lambda3|
        Rb2 = divide(lambda1, lam23, lam23)      # Rb = |lambda1|/sqrt(|lambda2||lambda3|)

        # Compute vesselness function
        Ra2 *= alpha
        Rb2 *= beta
        S2  *= (-2.0/S2.max()) if c is None else c
        sub(1.0, exp(Ra2, out=Ra2), out=Ra2) # 1-exp(-Ra^2/(2*alpha^2))
        exp(         Rb2, out=Rb2)           #   exp(-Rb^2/(2*beta^2))
        sub(1.0, exp(S2,  out=S2 ), out=S2 ) # 1-exp(-S^2/(2*c^2))

        place(filtered, mask, multiply(multiply(Ra2, Rb2, Ra2), S2, Ra2))
        del lambda1, lambda2, lambda3, Ra2, Rb2, S2

        # Store pixels which are maximal
        if first: copyto(out, filtered)
        elif return_full:
            greater(filtered, out, out=mask)
            putmask(out, mask, filtered)
        else: maximum(filtered, out, out=out)
        if return_full:
            putmask(sigs, mask, sigma)
            putmask(vecx, mask, vx); putmask(vecy, mask, vy); putmask(vecz, mask, vz)
        
        first = False
        
    # Return output
    return (out,sigs,vecx,vecy,vecz) if return_full else out

def hessian3(im, sigma, Dxx, Dxy, Dxz, Dyy, Dyz, Dzz, tmp):
    """
    Filters the image with the 2nd derivatives of a Gaussian with parameter `sigma`. This
    approximates the 2nd derivatives of the image.
    
    `hessian3(im, sigma, Dxx, Dyy, Dzz, Dxy, Dxz, Dyz, tmp)`
    
    Inputs:
        im      the input image volume
        sigma   the sigma of the Gaussian kernel used

    Outputs are the 2nd derivatives multiplied by sigma^2:
        Dxx     with respect to x both times
        Dxy     with respect to x then y or y then x (equivalent)
        Dxz     with respect to x then z or z then x (equivalent)
        Dyy     with respect to y both times
        Dyz     with respect to y then z or z then y (equivalent)
        Dzz     with respect to z both times
        tmp     used for temporary storage
       
    The outputs form the matrix:
        | Dxx  Dxy  Dxz |
        |               |
        | Dxy  Dyy  Dyz |
        |               |
        | Dxz  Dyz  Dzz |
    
    Written by Jeffrey Bush (NCMIR, 2016)
    """
    # Note: this uses convolution with the 2nd derivative of a Gaussian filter instead of how the
    # MATLAB code did it with a single Gaussian filter (not a derivative of that) and then using
    # centered differences. Their method does not produce the same results as this, but this method
    # should be more accurate to the original paper even though the other method is about twice as
    # fast. Original method converted to Python below.
    from numpy import pi, sqrt, exp, arange
    from scipy.ndimage import correlate1d
    sigma = float(sigma)
    sigma2,p = sigma*sigma, round(3*sigma)
    i = arange(p, -p-1.0, -1.0)
    i2 = i*i
    Gn = exp(-i2/sigma2)           # kernel for dims that we are not differentiating
    G1,G2 = i*Gn, (i2-sigma2/2)*Gn # kernel for dims that we are taking 1st and 2nd derivatives of
    tmp2 = Dxy # use this one as a second temporary since we don't need it till the end
    
    # Outputs never differentiated along the x-axis
    correlate1d(im, Gn, 0, tmp2)
    correlate1d(tmp2, G1, 1, tmp); correlate1d(tmp, G1, 2, Dyz)
    correlate1d(tmp2, G2, 1, tmp); correlate1d(tmp, Gn, 2, Dyy)
    correlate1d(tmp2, Gn, 1, tmp); correlate1d(tmp, G2, 2, Dzz)
    
    # Outputs never differentiated along the y-axis
    correlate1d(im, Gn, 1, tmp2)
    correlate1d(tmp2, G1, 0, tmp); correlate1d(tmp, G1, 2, Dxz)
    correlate1d(tmp2, G2, 0, tmp); correlate1d(tmp, Gn, 2, Dxx)
    
    # Outputs never differentiated along the z-axis
    correlate1d(im, Gn, 2, tmp2)
    correlate1d(tmp2, G1, 0, tmp); correlate1d(tmp, G1, 1, Dxy)

    # Do all dims at once (requires the scaling) - Takes >10x longer
    #from numpy import pi, sqrt, exp, ogrid
    #from scipy.ndimage import correlate
    #p,sigma2 = round(3*sigma), sigma*sigma
    #x,y,z = ogrid[p:-p-1.0:-1.0, p:-p-1.0:-1.0, p:-p-1.0:-1.0]
    #x2,y2,z2 = x*x, y*y, z*z
    #K = exp(-(x2+y2+z2)/sigma2)
    #correlate(im, K*(x2-sigma2/2), Dxx)
    #correlate(im, K*(y2-sigma2/2), Dyy)
    #correlate(im, K*(z2-sigma2/2), Dzz)
    #correlate(im, K*x*y, Dxy)
    #correlate(im, K*x*z, Dxz)
    #correlate(im, K*y*z, Dyz)

    # Scale all
    c = 2/(sqrt(2*(pi*pi*pi))*(sigma2*sigma2*sigma))
    Dxx *= c; Dyy *= c; Dzz *= c; Dxy *= c; Dxz *= c; Dyz *= c

# Original hessian3 method using central differences
#def hessian3(im, sigma, Dxx, Dyy, Dzz, Dzy, Dxz, Dyz, t):
#    from scipy.ndimage import gaussian_filter
#    F = gaussian_filter(im, sigma) if sigma > 0 else im
#    __gradient3(F, 0, t); __gradient3(t, 0, Dxx); __gradient3(t, 1, Dxy); __gradient3(t, 2, Dxz)
#    __gradient3(F, 1, t); __gradient3(t, 1, Dyy); __gradient3(t, 2, Dyz)
#    __gradient3(F, 2, t); __gradient3(t, 2, Dzz)
#def  __gradient3(F, axis, out):
#    from numpy import subtract
#    L = F.shape[axis]
#    A, B = (slice(None),)*axis, (slice(None),)*(2-axis)
#    frst, last = A + (0,) + B, A + (L-1,) + B
#    scnd, slst = A + (1,) + B, A + (L-2,) + B
#    rght, left = A + (slice(2,L-1),) + B, A + (slice(0,L-3),) + B
#    cntr = A + (slice(1,L-2),) + B
#    # Take forward differences on left and right edges
#    subtract(F[scnd], F[frst], out[frst])
#    subtract(F[last], F[slst], out[last])
#    # Take centered differences on interior points
#    subtract(F[rght], F[left], out[cntr]); out[cntr] *= 0.5

def eig3volume(Dxx, Dxy, Dxz, Dyy, Dyz, Dzz, return_vec=False):
    """
    Calculate the eigenvalues and possibly the eigenvectors from the Hessian matrix of an image
    volume, sorted by absolute value. Calculating the minimal eigenvector takes about an additional
    250% in time.
    
    `eig3volume(Dxx, Dxy, Dxz, Dyy, Dyz, Dzz, return_vec=False)`
    
    Inputs:
        Dxx, Dxy, Dxz, Dyy, Dyz, Dzz the outputs from hessian3
        return_vec       if True, the eigenvectors will be calculated as well (default is False)
    
    Outputs:
        Dxx, Dxy, Dxz    the eigenvalues, sorted by absolute value 
        Dyy, Dyz, Dzz    the eigenvector with the eigenvalue closest to zero (if calculated)
    
    Written by Jeffrey Bush (NCMIR, 2016)
    """
    # 1.5x to 3.5x as slow as the Cython code
    # 7 TEMPs at max with return_vec=False, and at least 21 TEMPs with return_vec=True (possibly more)

    from numpy import empty, arccos, cos, pi, sqrt, copyto
    from numpy import negative as neg, multiply as mul
    
    T = empty(Dxx.shape) # 1 TEMP
    
    q = Dxx + Dyy; q += Dzz; q /= 3    # 1 TEMP     # q = trace(A)/3
    Dxx -= q; Dyy -= q; Dzz -= q                    # A-q*I
    p2 = Dxx*Dxx; p2 += mul(Dyy,Dyy,T); p2 += mul(Dzz,Dzz,T) # 1 TEMP
    Dxy2 = Dxy*Dxy; Dxz2 = Dxz*Dxz; Dyz2 = Dyz*Dyz           # 3 TEMPs
    p = Dxy2 + Dxz2; p += Dyz2; p *= 2                       # 1 TEMP
    p2 += p; p2 /= 6.0; sqrt(p2, p)                 # p = ||A-q*I||/sqrt(6)
    
    # Calculate |(A-q*I)/p|/2
    Dxz2 -= mul(Dxx,Dzz,T)
    r = mul(Dxy,Dxz,T); r *= Dyz; r *= 2
    Dxz2 *= Dyy; r -= Dxz2
    Dxy2 *= Dzz; r -= Dxy2
    Dyz2 *= Dxx; r -= Dyz2
    r /= p2; r /= p; r *= 0.5
    del p2, Dxy2, Dxz2, Dyz2, T
    
    # phi = acos(r)/3 with r in [-1,1]
    phi = arccos(r.clip(-1, 1, r), r); phi /= 3 

    # Calculate eigenvalues: q+2*p*cos(phi+2*k*pi/3) with k=-1,0,1
    eigs = empty(Dxx.shape + (3,))  # 3 TEMPs
    cos(phi, eigs[:,:,:,2])
    phi += 2*pi/3
    cos(phi, eigs[:,:,:,0])
    neg(eigs[:,:,:,0], eigs[:,:,:,1]); eigs[:,:,:,1] -= eigs[:,:,:,2]
    p *= 2; eigs *= p[:,:,:,None]; eigs += q[:,:,:,None]
    del phi, p, r
        
    # Sort the eigenvalues by absolute value
    from numpy import ogrid
    inds = ogrid[:eigs.shape[0], :eigs.shape[1], :eigs.shape[2], :1]
    inds[3] = abs(eigs).argsort(3)
    eigs = eigs[inds]
    
    if return_vec:
        from numpy import finfo, float64, place, extract
        from numpy import absolute, logical_not, less_equal as le, subtract as sub
        
        Dxx_orig, Dxy_orig, Dxz_orig, Dyy_orig, Dzz_orig, Dyz_orig = Dxx, Dxy, Dxz, Dyy, Dzz, Dyz
        
        # Reset the diagonal
        Dxx += q; Dyy += q; Dzz += q
        del q, inds
        
        # Get the eigenvalues and the amount of error
        r,p,q = eigs[:,:,:,0], eigs[:,:,:,1], eigs[:,:,:,2]
        eps = abs(r); eps *= 1e6*finfo(float64).eps # 1 TEMP

        # Find the eigenvalues that are along the diagonal and are the only non-zero value in the row-column
        T, B = empty(Dxx.shape), empty(Dxx.shape, bool) # 2 TEMPs
        vec = [None, None, None]
        for i,(X,Y,Z) in enumerate(((Dxx, Dxy, Dxz), (Dyy, Dxy, Dyz), (Dzz, Dxz, Dyz))):
            # Generates 4 TEMPs and uses 1 TEMP extra
            X -= r
            msk = absolute(X, T) <= eps # 1 TEMP
            msk &= le(absolute(Y, T), eps, B)
            msk &= le(absolute(Z, T), eps, B)
            vec[i] = msk.astype(float64)
            if i == 0: mask  = msk
            else:      mask |= msk
            del msk
            X += r
        vec_x,vec_y,vec_z = vec
        del vec, r, T, B
        
        # Find the other eigenvalues by looking at the columns of (A-p*I)*(A-q*I)
        # Each time through we eliminate all elementes already completed
        remaining = logical_not(mask, mask)
        DxxP = Dxx-p; DxxQ = sub(Dxx, q, Dxx) # 1 TEMP
        DyyP = Dyy-p; DyyQ = sub(Dyy, q, Dyy) # 1 TEMP
        DzzP = Dzz-p; DzzQ = sub(Dzz, q, Dzz) # 1 TEMP
        for i in xrange(3):
            DxxP,DyyP,DzzP = extract(mask, DxxP), extract(mask, DyyP), extract(mask, DzzP) # 3 TEMPs
            DxxQ,DyyQ,DzzQ = extract(mask, DxxQ), extract(mask, DyyQ), extract(mask, DzzQ) # 3 TEMPs
            Dxy, Dxz, Dyz  = extract(mask, Dxy),  extract(mask, Dxz),  extract(mask, Dyz)  # 3 TEMPs
            eps = extract(mask, eps) # 1 TEMP
            if i == 0:
                T = empty(DxxP.shape) # 1 TEMP
                vx = DxxP*DxxQ; vx += mul(Dxy,Dxy,T); vx += mul(Dxz,Dxz,T) # 1 TEMP
                vy = DxxP+DyyQ; vy *= Dxy; vy += mul(Dyz,Dxz,T)            # 1 TEMP
                vz = DxxP+DzzQ; vz *= Dxz; vz += mul(Dyz,Dxy,T)            # 1 TEMP
                del T
            elif i == 1: # Others not optimized as well since they are unlikely to have many values computed in them
                vx,vy,vz = Dxy*(DxxP+DyyQ)+Dyz*Dxz, Dxy*Dxy+DyyP*DyyQ+Dyz*Dyz, Dxz*Dxy+Dyz*(DyyP+DzzQ) # 13 TEMPs
            else:
                vx,vy,vz = Dxz*(DxxP+DzzQ)+Dyz*Dxy, Dxz*Dxy+Dyz*(DyyP+DzzQ), Dxz*Dxz+Dyz*Dyz+DzzP*DzzQ # 13 TEMPs
            place(vec_x, remaining, vx); place(vec_y, remaining, vy); place(vec_z, remaining, vz)
            mask = absolute(vx,vx) <= eps; mask &= absolute(vy,vy) <= eps; mask &= absolute(vz,vz) <= eps # 3 TEMPs
            place(remaining, remaining, mask)
            del vx, vy, vz
            if remaining.sum() == 0: break
            elif i == 2: raise Exception('Failed to calculate eigenvalues')
                
        # Normalize the vectors
        N = vec_x*vec_x; N += vec_y*vec_y; N += vec_z*vec_z; sqrt(N, N) # 3 TEMPs
        vec_x /= N; vec_y /= N; vec_z /= N

        copyto(Dxx_orig, eigs[:,:,:,0]); copyto(Dxy_orig, eigs[:,:,:,1]); copyto(Dxz_orig, eigs[:,:,:,2])
        copyto(Dyy_orig, vec_x); copyto(Dyz_orig, vec_y); copyto(Dzz_orig, vec_z)
    else:
        copyto(Dxx, eigs[:,:,:,0]); copyto(Dxy, eigs[:,:,:,1]); copyto(Dxz, eigs[:,:,:,2])

# This can be done with eigh/eigvalsh as well, but it is 4-5x slower (since it is generalized to any matrix size)!
#def eig3volume(Dxx, Dxy, Dxz, Dyy, Dyz, Dzz, return_vec=False):
#    # 12 or 18 TEMPs
#    from numpy import stack, ogrid, copyto
#    from numpy.linalg import eigvalsh, eigh
#
#    # Compute the eigenvalues/eigenvectors
#    H = stack((Dxx,Dxy,Dxz, Dxy,Dyy,Dyz, Dxz,Dyz,Dzz), -1).reshape(Dxx.shape+(3,3)) # full Hermitian matrix from the parts 
#    if return_vec: eigs,vecs = eigh(H)
#    else:          eigs = eigvalsh(H)
#    del H
#
#    # Sort the eigenvalues by absolute value
#    inds = ogrid[:Dxx.shape[0], :Dxx.shape[1], :Dxx.shape[2], :1]
#    inds[3] = abs(eigs).argsort(3)
#    eigs = eigs[inds]
#    copyto(Dxx, eigs[:,:,:,0])
#    copyto(Dxy, eigs[:,:,:,1])
#    copyto(Dxz, eigs[:,:,:,2])
#    if return_vec:
#        vec_inds = ogrid[:Dxx.shape[0], :Dxx.shape[1], :Dxx.shape[2], :3, :1]
#        vec_inds[4] = inds[3][:,:,:,None,0:1]
#        vecs = vecs[vec_inds]
#        copyto(Dyy, vecs[:,:,:,0,0]); copyto(Dyz, vecs[:,:,:,1,0]); copyto(Dzz, vecs[:,:,:,2,0])
