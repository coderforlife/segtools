#!python
#cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
#distutils: language=c++
#
# Cython helper module for Frangi filters.
# These are not required but are faster and use much less memory then the pure-Python ones.

from __future__ import division

from pysegtools.general.cython.npy_helper cimport *
import_array()

from libc.math cimport floor, fabs, sqrt, exp, atan2, acos, cos, M_PI
from libc.float cimport DBL_EPSILON

__all__ = ['frangi2', 'frangi3']

with_cython = True

ctypedef npy_double dbl
ctypedef dbl* dbl_p

########## Frangi 2D ##########
def frangi2(ndarray im, ndarray out, tuple sigmas, dbl beta, dbl c, bint black, bint return_full):
    """
    Internal function to compute the 2D Frangi filter according to Frangi et al (1998).
    
    See frangi.frangi2 for more details. A few differences in the signature are that all arguments
    are required and a value of 0.0 is used for c to indicate a dynamic value.

    Written by Jeffrey Bush (NCMIR, 2016)
    Adapted from the MATLAB version by M. Schrijver (2001) and D. Kroon (2009)
    """
    # Allocate arrays: 4 temporaries and possibly 3 outputs
    cdef ndarray Dxx, Dxy, Dyy, tmp, sigs, dirs
    Dxx = PyArray_EMPTY(2, PyArray_SHAPE(im), NPY_DOUBLE, False)
    Dxy = PyArray_EMPTY(2, PyArray_SHAPE(im), NPY_DOUBLE, False)
    Dyy = PyArray_EMPTY(2, PyArray_SHAPE(im), NPY_DOUBLE, False)
    tmp = PyArray_EMPTY(2, PyArray_SHAPE(im), NPY_DOUBLE, False)
    cdef dbl_p pDxx = <dbl_p>PyArray_DATA(Dxx), pDxy = <dbl_p>PyArray_DATA(Dxy), pDyy = <dbl_p>PyArray_DATA(Dyy)
    cdef dbl_p lambda1 = pDxx, lambda2 = pDxy, vx, vy
    cdef dbl_p pOut = <dbl_p>PyArray_DATA(out), pSigs, pDirs
    if return_full:
        sigs = PyArray_ZEROS(2, PyArray_SHAPE(im), NPY_DOUBLE, False)
        dirs = PyArray_ZEROS(2, PyArray_SHAPE(im), NPY_DOUBLE, False)
        pSigs = <dbl_p>PyArray_DATA(sigs); pDirs = <dbl_p>PyArray_DATA(dirs)
        vx = pDyy; vy = <dbl_p>PyArray_DATA(tmp)
  
    # Frangi filter for all sigmas
    cdef dbl sigma, lam1, lam2, Rb2, S2, V, b = 1 if black else -1
    beta = -1.0/(2.0*beta*beta)
    cdef bint c_dyn = c == 0.0
    if not c_dyn: c = -1.0/(2.0*c*c)
    cdef intp i, N = PyArray_SIZE(im)
    cdef bint first = True
    for sigma in sigmas:
        # Calculate the scaled 2D hessian
        hessian2(im, sigma, Dxx, Dxy, Dyy, tmp)
        
        with nogil:
            # Calculate (abs sorted) eigenvalues and eigenvectors
            if return_full: eig2image(N, pDxx, pDxy, pDyy, vy)
            else: eigval2image(N, pDxx, pDxy, pDyy)
            
            # Calculate the dynamic c value if necessary
            if c_dyn:
                c = 0.0
                for i in xrange(N):
                    lam2 = lambda2[i]
                    if b*lam2 > 0.0:
                        lam1 = lambda1[i]
                        S2 = lam1*lam1 + lam2*lam2
                        if S2 > c: c = S2
                c = -2.0/c
            
            # Calculate the vesselness
            for i in xrange(N):
                lam2 = lambda2[i]
                if b*lam2 > 0.0:
                    lam1 = lambda1[i]
                    
                    # Compute similarity measures
                    lam1 *= lam1; lam2 *= lam2
                    Rb2 =     exp(lam1/lam2*beta) # exp(-Rb^2/(2*beta^2)); Rb = lambda1 / lambda2
                    S2  = 1.0-exp((lam1+lam2)*c)  # 1-exp(-S^2/(2*c^2));   S = sqrt(sum(lambda_i^2))
                    V = Rb2 * S2

                    # If maximal, store values
                    if V > pOut[i]:
                        pOut[i] = V
                        if return_full:
                            pSigs[i] = sigma
                            pDirs[i] = atan2(vy[i], vx[i])
                    elif first: pOut[i] = 0
                    
            first = False

    # Return output
    return (out,sigs,dirs) if return_full else out

cdef void hessian2(ndarray im, dbl sigma, ndarray Dxx, ndarray Dxy, ndarray Dyy, ndarray tmp):
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
    # 10x faster, especially since Scipy never tries to separate the filters itself). Also, the
    # results are multipled by sigma^2 instead of doing that in the frangi2 code instead to save
    # computation.
    #
    # Not any faster or memory efficent than the Python version...
    from numpy import exp
    from scipy.ndimage import correlate1d
    
    cdef dbl sigma2 = sigma*sigma, p = floor(3.0*sigma+0.5)
    cdef ndarray i = PyArray_Arange(p, -p-1.0, -1.0, NPY_DOUBLE), i2 = i*i
    cdef ndarray v = exp(-i2/(2.0*sigma2)) * (1.0/(sqrt(2.0*M_PI)*sigma2))
    cdef ndarray uu = v*i,u = v*(i2-sigma2)
    correlate1d(im, uu, 0, tmp); correlate1d(tmp, uu, 1, Dxy)
    correlate1d(im, u,  0, tmp); correlate1d(tmp, v,  1, Dxx)
    correlate1d(im, v,  0, tmp); correlate1d(tmp, u,  1, Dyy)

cdef void eigval2image(intp N, dbl_p Dxx, dbl_p Dxy, dbl_p Dyy) nogil:
    """
    Calculate the eigenvalues from the Hessian matrix of an image, sorted by absolute value.

    `eigval2image(N,Dxx,Dxy,Dyy)`
    
    Inputs:
        N               number of elements in the arrays
        Dxx,Dxy,Dyy     the outputs from hessian2
        
    Outputs:
        Dxx,Dxy         the eigenvalues, sorted by absolute value
    """
    # About 2.5x faster than Python version
    cdef dbl summ, diff, tmp, mu1, mu2
    cdef intp i
    for i in xrange(N):
        summ = Dxx[i] + Dyy[i]
        diff = Dxx[i] - Dyy[i]
        tmp = sqrt(diff*diff + 4*Dxy[i]*Dxy[i])
        
        # Compute the eigenvalues
        mu1 = 0.5*(summ + tmp) # mu1 = (Dxx + Dyy + sqrt((Dxx-Dyy)^2+4*Dxy^2)) / 2
        mu2 = 0.5*(summ - tmp) # mu2 = (Dxx + Dyy - sqrt((Dxx-Dyy)^2+4*Dxy^2)) / 2
        
        # Sort eigenvalues by absolute value abs(eig1) < abs(eig2)
        if fabs(mu1) > fabs(mu2): Dxx[i] = mu2; Dxy[i] = mu1
        else:                     Dxx[i] = mu1; Dxy[i] = mu2

cdef void eig2image(intp N, dbl_p Dxx, dbl_p Dxy, dbl_p Dyy, dbl_p vy) nogil:
    """
    Calculate the eigenvalues and eigenvectors from the Hessian matrixof an image, sorted by
    absolute value.
    
    `eig2image(N,Dxx,Dxy,Dyy,vy)`
    
    Inputs:
        N               number of elements in the arrays
        Dxx,Dxy,Dyy     the outputs from hessian2
        
    Outputs:
        Dxx,Dxy         the eigenvalues, sorted by absolute value
        Dyy,vy          the eigenvector with the eigenvalue closest to zero
    """
    
    # About 2.5x faster than Python version
    cdef dbl summ, diff, tmp, mag, mu1, mu2, v2x, v2y
    cdef intp i
    for i in xrange(N):
        summ = Dxx[i] + Dyy[i]
        diff = Dxx[i] - Dyy[i]
        tmp = sqrt(diff*diff + 4.0*Dxy[i]*Dxy[i])

        # Compute the eigenvalues
        mu1 = 0.5*(summ + tmp) # mu1 = (Dxx + Dyy + sqrt((Dxx-Dyy)^2+4*Dxy^2)) / 2
        mu2 = 0.5*(summ - tmp) # mu2 = (Dxx + Dyy - sqrt((Dxx-Dyy)^2+4*Dxy^2)) / 2

        # Compute and normalize the eigenvectors
        v2x = 2.0*Dxy[i] # v2x = 2 * Dxy
        v2y = tmp - diff # v2y = sqrt((Dxx-Dyy)^2+4*Dxy^2)) - Dxx + Dyy
        if v2x != 0.0 and v2y != 0.0:
            mag = sqrt(v2x*v2x + v2y*v2y)
            v2x /= mag
            v2y /= mag

        # Sort eigenvalues by absolute value abs(eig1) < abs(eig2)
        # Also, the eigenvectors are orthogonal
        if fabs(mu1) > fabs(mu2):
            Dxx[i] = mu2; Dxy[i] = mu1; Dyy[i] = v2y; vy[i] =  v2x
        else:
            Dxx[i] = mu1; Dxy[i] = mu2; Dyy[i] = v2x; vy[i] = -v2y


########## Frangi 3D ##########
def frangi3(ndarray im, ndarray out, tuple sigmas, dbl alpha, dbl beta, dbl c, bint black, bint return_full):
    """
    Internal function to compute the 3D Frangi filter according to Frangi et al (1998).
    
    See frangi.frangi3 for more details. A few differences in the signature are that all arguments
    are required and a value of 0.0 is used for c to indicate a dynamic value.
    
    Written by Jeffrey Bush (NCMIR, 2016)
    Adapted from the MATLAB version by D. Kroon (2009)
    """
    # Allocate arrays: 7 temporaries and possibly 4 outputs
    cdef ndarray Dxx, Dxy, Dxz, Dyy, Dyz, Dzz, tmp, sigs, vecx, vecy, vecz
    Dxx = PyArray_EMPTY(3, PyArray_SHAPE(im), NPY_DOUBLE, False)
    Dxy = PyArray_EMPTY(3, PyArray_SHAPE(im), NPY_DOUBLE, False)
    Dxz = PyArray_EMPTY(3, PyArray_SHAPE(im), NPY_DOUBLE, False)
    Dyy = PyArray_EMPTY(3, PyArray_SHAPE(im), NPY_DOUBLE, False)
    Dyz = PyArray_EMPTY(3, PyArray_SHAPE(im), NPY_DOUBLE, False)
    Dzz = PyArray_EMPTY(3, PyArray_SHAPE(im), NPY_DOUBLE, False)
    tmp = PyArray_EMPTY(3, PyArray_SHAPE(im), NPY_DOUBLE, False)
    cdef dbl_p pDxx = <dbl_p>PyArray_DATA(Dxx), pDxy = <dbl_p>PyArray_DATA(Dxy), pDxz = <dbl_p>PyArray_DATA(Dxz)
    cdef dbl_p pDyy = <dbl_p>PyArray_DATA(Dyy), pDyz = <dbl_p>PyArray_DATA(Dyz), pDzz = <dbl_p>PyArray_DATA(Dzz)
    cdef dbl_p lambda1 = pDxx, lambda2 = pDxy, lambda3 = pDxz, vx, vy, vz
    cdef dbl_p pOut = <dbl_p>PyArray_DATA(out), pSigs, pVecX, pVecY, pVecZ
    if return_full:
        sigs = PyArray_ZEROS(3, PyArray_SHAPE(im), NPY_DOUBLE, False)
        vecx = PyArray_ZEROS(3, PyArray_SHAPE(im), NPY_DOUBLE, False)
        vecy = PyArray_ZEROS(3, PyArray_SHAPE(im), NPY_DOUBLE, False)
        vecz = PyArray_ZEROS(3, PyArray_SHAPE(im), NPY_DOUBLE, False)
        pSigs = <dbl_p>PyArray_DATA(sigs); pVecX = <dbl_p>PyArray_DATA(vecx)
        pVecY = <dbl_p>PyArray_DATA(vecy); pVecZ = <dbl_p>PyArray_DATA(vecz)
        vx = pDyy; vy = pDyz; vz = pDzz

    # Frangi filter for all sigmas
    cdef dbl sigma, lam1, lam2, lam3, lam23, Ra2, Rb2, S2, V, b = 1 if black else -1
    alpha = -1.0/(2.0*alpha*alpha)
    beta = -1.0/(2.0*beta*beta)
    cdef bint c_dyn = c == 0.0
    if not c_dyn: c = -1.0/(2.0*c*c)
    cdef intp i, N = PyArray_SIZE(im)
    cdef bint first = True
    for sigma in sigmas:
        # Calculate scaled 3D Hessian
        hessian3(im, sigma, Dxx, Dxy, Dxz, Dyy, Dyz, Dzz, tmp)
    
        with nogil:
            # Calculate (abs sorted) eigenvalues and eigenvectors
            eig3volume(N, pDxx, pDxy, pDxz, pDyy, pDyz, pDzz, return_full)
            
            # Calculate the dynamic c value if necessary
            if c_dyn:
                c = 0.0
                for i in xrange(N):
                    lam2 = lambda2[i]; lam3 = lambda3[i]
                    if lam2 != 0.0 and lam3 != 0.0 and (b*lam2 > 0.0 or b*lam3 > 0.0):
                        lam1 = lambda1[i]
                        S2 = lam1*lam1 + lam2*lam2 + lam3*lam3
                        if S2 > c: c = S2
                c = -2.0/c

            # Calculate the vesselness
            for i in xrange(N):
                lam2 = lambda2[i]; lam3 = lambda3[i]
                if lam2 != 0.0 and lam3 != 0.0 and (b*lam2 > 0.0 or b*lam3 > 0.0):
                    lam1 = lambda1[i]
                    
                    # Ra2/Rb2 -> vesselness features
                    # S -> second order structureness
                    lam23 = fabs(lam2*lam3); lam1 *= lam1; lam2 *= lam2; lam3 *= lam3
                    Ra2 = 1.0-exp(lam2/lam3*alpha)    # 1-exp(-Ra^2/(2*alpha^2)); Ra = |lambda2|/|lambda3|
                    Rb2 =     exp(lam1/lam23*beta)    #   exp(-Rb^2/(2*beta^2));  Rb = |lambda1|/sqrt(|lambda2||lambda3|)
                    S2  = 1.0-exp((lam1+lam2+lam3)*c) # 1-exp(-S^2/(2*c^2));      S = sqrt(sum(lambda_i^2))
                    V = Ra2*Rb2*S2
                    
                    # If maximal, store values
                    if V > pOut[i]:
                        pOut[i] = V
                        if return_full:
                            pSigs[i] = sigma
                            pVecX[i] = vx[i]; pVecY[i] = vy[i]; pVecZ[i] = vz[i]
                    elif first: pOut[i] = 0
            
            first = False

    return (out,sigs,vecx,vecy,vecz) if return_full else out
    
cdef void hessian3(ndarray im, dbl sigma, ndarray Dxx, ndarray Dxy, ndarray Dxz, ndarray Dyy, ndarray Dyz, ndarray Dzz, ndarray tmp):
    """
    Filters an image with the 2nd derivatives of a Gaussian with parameter `sigma`. This
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
    # fast. Original method converted to Python is in Python code.
    #
    # Not any faster or memory efficent than the Python version...
    from numpy import exp
    from scipy.ndimage import correlate1d
    
    cdef dbl sigma2 = sigma*sigma, p = round(3.0*sigma+0.5)
    cdef ndarray i = PyArray_Arange(p, -p-1.0, -1.0, NPY_DOUBLE), i2 = i*i
    
    cdef ndarray Gn = exp(-i2/sigma2)           # kernel for dims that we are not differentiating
    cdef ndarray G1=i*Gn, G2=(i2-sigma2/2.0)*Gn # kernel for dims that we are taking 1st and 2nd derivatives of
    cdef ndarray tmp2 = Dxy # use this one as a second temporary since we don't need it till the end

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

    # Scale all
    c = 2.0/(sqrt(2.0*M_PI*M_PI*M_PI)*(sigma2*sigma2*sigma))
    Dxx *= c; Dyy *= c; Dzz *= c; Dxy *= c; Dxz *= c; Dyz *= c

cdef bint eig3volume(intp N, dbl_p Dxx, dbl_p Dxy, dbl_p Dxz, dbl_p Dyy, dbl_p Dyz, dbl_p Dzz, bint return_vec=False) nogil:
    """
    Calculate the eigenvalues and possibly the eigenvectors from the Hessian matrix of an image
    volume, sorted by absolute value. Calculating the minimal eigenvector only takes about an
    additional 20% in time.
    
    `eig3volume(N, Dxx, Dxy, Dxz, Dyy, Dyz, Dzz, return_vec=False)`
    
    Inputs:
        N                the number of elements in each of the arrays
        Dxx, Dxy, Dxz, Dyy, Dyz, Dzz the outputs from hessian3
        return_vec       if True, the eigenvectors will be calculated as well (default is False)
    
    Outputs:
        Dxx, Dxy, Dxz    the eigenvalues, sorted by absolute value 
        Dyy, Dyz, Dzz    the eigenvector with the eigenvalue closest to zero (if calculated)
        (return value)   if non-zero, the results failed to be computed
        
    Written by Jeffrey Bush (NCMIR, 2016)
    """
    cdef intp i
    cdef int retval
    cdef dbl[3] eigs, vec_temp # temporary eigenvalues and eigenvectors
    cdef dbl_p vec = <dbl_p>(vec_temp if return_vec else NULL)
    for i in xrange(N):
        # Compute the eigenvalues/eigenvectors
        retval = eig_calc(Dxx[i], Dxy[i], Dxz[i], Dyy[i], Dyz[i], Dzz[i], eigs, vec)
        if retval != 0: break
        # Store values
        Dxx[i] = eigs[0]; Dxy[i] = eigs[1]; Dxz[i] = eigs[2]
        if return_vec: Dyy[i] = vec[0]; Dyz[i] = vec[1]; Dzz[i] = vec[2]
    return retval

cdef bint eig_calc(dbl Dxx, dbl Dxy, dbl Dxz, dbl Dyy, dbl Dyz, dbl Dzz, dbl[3] eigs, dbl[3] vec) nogil:
    """
    Internal function that calculates a single set of eigenvalues and possibly the minimal
    eigenvector for a 3x3 real symmetric matrix.
    
    Written by Jeffrey Bush (NCMIR, 2016)
    """
    # Calculate eignvalues
    cdef dbl A11 = Dxx, A12 = Dxy, A13 = Dxz, A22 = Dyy, A23 = Dyz, A33 = Dzz
    cdef dbl tr, p2, p, q, r, phi #, eig1, eig2, eig3
    p = A12*A12 + A13*A13 + A23*A23
    if p == 0.0:
        # Diagonal matrix; eigenvalues are unsorted
        eigs[0] = A11; eigs[1] = A22; eigs[2] = A33
    else:
        tr = A11 + A22 + A33 # trace(A)
        q = tr/3.0
        A11 -= q; A22 -= q; A33 -= q
        p2 = (A11*A11 + A22*A22 + A33*A33 + 2.0*p) / 6.0
        p = sqrt(p2)         # p = ||A-qI||/sqrt(6)
        r = A11*(A22*A33-A23*A23) - A12*(A12*A33-A23*A13) + A13*(A12*A23-A22*A13)
        r *= 1.0/(2.0*p2*p)  # r = |A-qI|/(2*p^3)
        # Due to rounding errors r may be just outside [-1, 1] which is invalid for the acos function
        if   r <= -1.0: r = -1.0
        elif r >=  1.0: r = 1.0
        phi = acos(r)/3.0
        # Eigenvalues are sorted in ascending order
        eigs[2] = q + 2.0*p*cos(phi)
        eigs[0] = q + 2.0*p*cos(phi + (2.0*M_PI/3.0))
        eigs[1] = tr - eigs[0] - eigs[2] # trace(A) = eig1 + eig2 + eig3

    # Sort eigenvalues by absolute value
    cdef dbl* eigs_abs = [fabs(eigs[0]), fabs(eigs[1]), fabs(eigs[2])]
    if eigs_abs[1] < eigs_abs[0]:
        p = eigs[0]
        if   eigs_abs[2] < eigs_abs[1]: eigs[0] = eigs[2]; eigs[2] = p
        elif eigs_abs[2] < eigs_abs[0]: eigs[0] = eigs[1]; eigs[1] = eigs[2]; eigs[2] = p
        else:                           eigs[0] = eigs[1]; eigs[1] = p
    elif eigs_abs[2] < eigs_abs[1]:
        p = eigs[2]; eigs[2] = eigs[1]
        if eigs_abs[2] < eigs_abs[0]: eigs[1] = eigs[0]; eigs[0] = p
        else:                         eigs[1] = p

    # Check if calculating vectors
    if vec is NULL: return False
    A11 = Dxx; A22 = Dyy; A33 = Dzz # reset these values, the eigenvalue calculations changes them
    
    # Calculate eigenvectors
    # Check if the minimum eigenvalue is along the diagonal and the only non-zero in its row/column
    # If so, the corresponding eigenvector is one of [1,0,0], [0,1,0], or [0,0,1]
    r = eigs[0]
    cdef dbl eps = (1.0e6*DBL_EPSILON)*fabs(r)
    if fabs(A11-r) <= eps and fabs(A12) <= eps and fabs(A13) <= eps: vec[0] = 1.0; vec[1] = vec[2] = 0.0; return False
    if fabs(A22-r) <= eps and fabs(A12) <= eps and fabs(A23) <= eps: vec[1] = 1.0; vec[0] = vec[2] = 0.0; return False
    if fabs(A33-r) <= eps and fabs(A13) <= eps and fabs(A23) <= eps: vec[2] = 1.0; vec[0] = vec[1] = 0.0; return False
    # Calculate the columns of (A-pI)*(A-qI) where p and q are the non-minimal eigenvalues and if a column
    # isn't all zeros then it is the eigenvector for the minimal eigenvalue. Note that (A-pI)*(A-qI) is symmetric.
    p = eigs[1]; q = eigs[2]; p2 = p+q
    return (ev((A11-p)*(A11-q)+A12*A12+A13*A13, A12*(A11+A22-p2)+A23*A13, A13*(A11+A33-p2)+A23*A12, vec, eps) and
            ev(A12*(A11+A22-p2)+A23*A13, A12*A12+(A22-p)*(A22-q)+A23*A23, A13*A12+A23*(A22+A33-p2), vec, eps) and
            ev(A13*(A11+A33-p2)+A23*A12, A13*A12+A23*(A22+A33-p2), A13*A13+A23*A23+(A33-p)*(A33-q), vec, eps))

cdef inline bint ev(dbl Vx, dbl Vy, dbl Vz, dbl[3] vec, dbl eps) nogil:
    if fabs(Vx) <= eps and fabs(Vy) <= eps and fabs(Vz) <= eps: return True
    cdef dbl N = sqrt(Vx*Vx + Vy*Vy + Vz*Vz)
    vec[0] = Vx / N; vec[1] = Vy / N; vec[2] = Vz / N
    return False

# This can be done with LAPACK as well, but it is 6-13x slower (since it is generalized to any matrix size)!
#from scipy.linalg.cython_lapack cimport dspev
#cdef int eig_calc(dbl Dxx, dbl Dxy, dbl Dxz, dbl Dyy, dbl Dyz, dbl Dzz, dbl[3] eigs, dbl[3] vec) nogil:
#    # Get the eigenvalues and eigenvectors with LAPACK
#    cdef int n = 3, retval
#    cdef dbl[6] A             # temporary 3x3 packed upper-triangular matrix
#    cdef dbl[9] vectors, work # temporary space for all 3 eigenvectors and working space
#    A[0] = Dxx
#    A[1] = Dxy; A[2] = Dyy
#    A[3] = Dxz; A[4] = Dyz; A[5] = Dzz
#    dspev('N' if vec is NULL else 'V', 'U', &n, A, eigs, vectors, &n, work, &retval)
#    if retval != 0: return retval
#
#    # Sort eigenvalues by absolute value
#    cdef dbl* eigs_abs = [fabs(eigs[0]), fabs(eigs[1]), fabs(eigs[2])]
#    cdef dbl t
#    cdef intp a = 0
#    if eigs_abs[1] < eigs_abs[0]:
#        t = eigs[0]
#        if   eigs_abs[2] < eigs_abs[1]: eigs[0] = eigs[2]; eigs[2] = t; a = 6
#        elif eigs_abs[2] < eigs_abs[0]: eigs[0] = eigs[1]; eigs[1] = eigs[2]; eigs[2] = t; a = 3
#        else:                           eigs[0] = eigs[1]; eigs[1] = t; a = 3
#    elif eigs_abs[2] < eigs_abs[1]:
#        t = eigs[2]; eigs[2] = eigs[1]
#        if eigs_abs[2] < eigs_abs[0]: eigs[1] = eigs[0]; eigs[0] = t; a = 6
#        else:                         eigs[1] = t
#            
#    # Save minimal eigenvector
#    if vec is not NULL: vec[0] = vectors[0+a]; vec[1] = vectors[1+a]; vec[2] = vectors[2+a]
#    
#    return 0
