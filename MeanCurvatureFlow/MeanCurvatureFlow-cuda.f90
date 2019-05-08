!*******************************************************************************
! Minimal surface generation by Mean curvature flow
!*******************************************************************************
program main
    implicit real*8 (a-h, o-z)
    real*8, allocatable :: ar2(:), adsep(:)
    real*8, allocatable :: g(:, :, :), phi(:, :, :)
    real*8, allocatable :: gx(:, :, :), gy(:, :, :), gz(:, :, :)

    !Computational domain
    xl = 0.0d0; xr = 5.5d0
    yl = -1.5d0; yr = 1.5d0
    zl = -1.5d0; zr = 1.5d0
    dx = 0.5d-1
    nx = anint((xr-xl)/dx) + 1; ny = anint((yr-yl)/dx) + 1; nz = anint((zr-zl)/dx) + 1
    write(*, *) "x", xl, xr, dx, nx
    write(*, *) "y", yl, yr, dx, ny
    write(*, *) "z", zl, zr, dx, nz
    
    t = 0.0d0
    tend = 1.0d1
    dt = 2.0d-4
    nt = anint(tend/dt) + 1
    write(*, *) "t", t, dt, tend
    nr2 = 1
    ndsep = 1
    allocate(ar2(nr2), adsep(ndsep), stat = ierr)
    do i = 1, nr2
        ar2(i)=1.0d0
    enddo
    adsep(1) = 0.4d0
    !adsep(2) = 0.1d0
    !adsep(3) = 0.2d0
    !adsep(4) = 0.3d0
    !adsep(5) = 0.4d0
    
    !Centers and radius of two balls
    r1 = 1.0d0
    c1x = 1.5d0; c1y = 0.0d0; c1z = 0.0d0
    c2y = 0.0d0; c2z = 0.0d0
    allocate(g(nx, ny, nz), phi(nx, ny, nz), stat = ierr)
    allocate(gx(nx, ny, nz), gy(nx, ny, nz), gz(nx, ny, nz), stat = ierr)
    
    do ir2 = 1, nr2
        r2 = ar2(ir2)
        do idsep = 1, ndsep
            dsep = adsep(idsep) 
            c2x = c1x + r1 + r2 + dsep
            ifort = 99
            write(*, *) r1, r2, dsep, ifort
            
            call initial(nx, ny, nz, xl, yl, zl, dx, c1x, c1y, c1z, c2x, c2y, c2z, r1, r2, g, phi, vmax, vmin)
            write(*, *) "Thresholding values:", vmin, vmax
            
            call gradient_g(nx, ny, nz, g, gx, gy, gz, dx)
            
            t = 0.0d0
            do it = 1, nt
  		 call cpu_time(start_time)
 	
                !original:call upwinding(nx, ny, nz, dx, dt, g, gx, gy, gz, phi)
                call upwinding(nx, ny, nz, dx, dt, g, gx, gy, gz, phi)
			
 		call cpu_time(stop_time) 
 		TIME = start_time - stop_time

                write(*, *) "Iteration", it, time
            enddo
            
            open(unit = 1, file = "fort.txt", status = "replace", action = "write", position = "rewind", iostat = nopenstatus)
                if (nopenstatus < 0) stop
                write(1, *) nz
                write(1, *) ny
                write(1, *) nx
                do iz = 1, nz
                    do iy = 1, ny
                        do ix = 1, nx
                            write(1, '(e15.5)') phi(ix, iy, iz)
                        enddo
                    enddo
                enddo
        enddo
    enddo

    !Output to dx to visualize the iso-surface, To render by VMD
    open(10, file = "TwoAtoms.dx")
        write(10, "('# Iosvalues of the two atoms system, ', $)")
        write(10, "('#')")
        write(10, "('object 1 class gridpositions counts 'I5, I5, I5)"), nx, ny, nz
        write(10, "('origin ' f9.3, x, f9.3,x, f9.3)"), xl, yl, zl
        write(10, "('delta  ' 3f8.3)"), dx, 0.0, 0.0
        write(10, "('delta  ' 3f8.3)"), 0.0, dx, 0.0
        write(10, "('delta  ' 3f8.3)"), 0.0, 0.0, dx
        write(10, "('object 2 class gridpositions counts 'I5, I5, I5)"), nx, ny, nz
        write(10, "('object 3 class array type double rank 0 items ' I8'  data follows')"), nx*ny*nz
        do i = 1, nx
            do j = 1, ny
                do k = 1, nz
                    write(10, *) phi(i, j, k)
                enddo
            enddo
        enddo
    close(10)
    
    deallocate(ar2, adsep, g, phi, gx, gy, gz, stat = ierr)     
end program main

!*******************************************************************************
! Initialization
!*******************************************************************************
subroutine initial(nx, ny, nz, xl, yl, zl, dx, c1x, c1y, c1z, c2x, c2y, c2z, r1, r2, g, phi, vmax, vmin)
    implicit real*8 (a-h, o-z)
    dimension g(nx, ny, nz), phi(nx, ny, nz), xi(nx), yi(ny), zi(nz)

    alpha = 1.0d3       !scaling factor of initial value
    eps = 1.0d-7
    vmin = 0.0d0        !fixed dirichlet bc
    phi = vmin
    vmax = alpha        !fixed edge values
    
    beta = 1.3d0
    do ix = 1, nx
        xi(ix) = xl + (ix - 1.0d0) * dx
    enddo
    do iy = 1, ny
        yi(iy) = yl + (iy - 1.0d0) * dx
    enddo
    do iz = 1, nz
        zi(iz) = zl + (iz - 1.0d0) * dx
    enddo
    
    r1a = r1*beta; r2a = r2*beta
    
    do ix = anint((c1x - r1a - xl)/dx + 1), anint((c1x + r1a - xl)/dx + 1)
        rx = sqrt(abs(r1a**2 - (xi(ix) - c1x)**2))
        do iy = anint((c1y - rx - yl)/dx + 1), anint((c1y + rx - yl)/dx + 1)
            dist = sqrt(abs(rx**2 - (yi(iy) - c1y)**2))
            do iz = anint((c1z - dist - zl)/dx + 1), anint((c1z + dist - zl)/dx + 1)
                phi(ix, iy, iz) = vmax
            enddo
        enddo
    enddo
    
    do ix = anint((c2x - r2a - xl)/dx + 1), anint((c2x + r2a - xl)/dx + 1)
        rx = sqrt(abs(r2a**2 - (xi(ix) - c2x)**2))
        do iy = anint((c2y - rx - yl)/dx + 1), anint((c2y + rx - yl)/dx + 1)
            dist = sqrt(abs(rx**2 - (yi(iy) - c2y)**2))
            do iz = anint((c2z - dist - zl)/dx + 1), anint((c2z + dist - zl)/dx + 1)
                phi(ix, iy, iz) = vmax
            enddo
        enddo
    enddo
    
    g = 1.0d0   !edge detector
    ixc = anint((c1x - xl)/dx + 1); iyc = anint((c1y - yl)/dx + 1); izc = anint((c1z - zl)/dx + 1)
    pi = dacos(-1.0d0)
    zeta0 = -pi/2.0d0; zeta1 = pi/2.0d0
    dzeta = dx/r1
    nzeta = anint((zeta1 - zeta0)/dzeta) + 11
    dzeta = (zeta1 - zeta0)/(nzeta - 1.0d0)
    
    do izeta = 1, nzeta
        zeta = zeta0 + (izeta - 1.0d00) * dzeta
        zp = c1z + r1*sin(zeta)
        iz = anint((zp-zl)/dx + 1)
        iz2 = iz - anint((iz - izc)/(abs(iz - izc) + eps))
        rz = sqrt(abs(r1**2 - (zp - c1z)**2))
        if(rz.le.eps)then
            rz = eps
        endif
        
        theta0 = 0.0d0; theta1 = 2.0d0 * pi
        dtheta = dx/rz
        ntheta = anint((theta1 - theta0)/dtheta) + 11
        dtheta = (theta1 - theta0)/(ntheta - 1.0d0)
        do itheta = 1, ntheta
            theta = theta0 + (itheta - 1.0d0) * dtheta
            xp = c1x + rz*cos(theta)
            yp = c1y + rz*sin(theta)
            iy = anint((yp - yl)/dx + 1)
            ix = anint((xp - xl)/dx + 1)
            ix2 = ix - anint((ix - ixc)/(abs(ix - ixc) + eps))
            iy2 = iy - anint((iy - iyc)/(abs(iy - iyc) + eps))
            g(ix, iy, iz) = 0.0d0
            phi(ix, iy, iz) = vmax
            g(ix2, iy, iz) = 0.0d0
            phi(ix2, iy, iz) = vmax
            g(ix, iy2, iz) = 0.0d0
            phi(ix, iy2, iz) = vmax
            g(ix, iy, iz2) = 0.0d0
            phi(ix, iy, iz2) = vmax
        enddo
    enddo
    
    ixc = anint((c2x - xl)/dx + 1); iyc = anint((c2y - yl)/dx + 1); izc = anint((c2z - zl)/dx + 1)
    zeta0 = -pi/2.0d0; zeta1 = pi/2.0d0
    dzeta = dx/r2
    nzeta = anint((zeta1 - zeta0)/dzeta) + 11
    dzeta = (zeta1 - zeta0)/(nzeta - 1.0d0)
    
    do izeta = 1, nzeta
        zeta = zeta0 + (izeta - 1.0d00) * dzeta
        zp = c2z + r2 * sin(zeta)
        iz = anint((zp - zl)/dx + 1)
        iz2 = iz - anint((iz - izc)/(abs(iz - izc) + eps))
        rz = sqrt(abs(r2**2 - (zp - c2z)**2))
        if(rz <= eps)then
            rz = eps
        endif
        
        theta0 = 0.0d0; theta1 = 2.0d0 * pi
        dtheta = dx/rz
        ntheta = anint((theta1 - theta0)/dtheta) + 11
        dtheta = (theta1 - theta0)/(ntheta - 1.0d0)
        do itheta = 1, ntheta
            theta = theta0 + (itheta - 1.0d0) * dtheta
            xp = c2x + rz * cos(theta); yp = c2y + rz * sin(theta)
            iy = anint((yp - yl)/dx +1); ix = anint((xp - xl)/dx +1)
            ix2 = ix - anint((ix - ixc)/(abs(ix - ixc) + eps))
            iy2 = iy - anint((iy - iyc)/(abs(iy - iyc) + eps))
            g(ix, iy, iz) = 0.0d0; phi(ix, iy, iz) = vmax
            g(ix2, iy, iz) = 0.0d0; phi(ix2, iy, iz) = vmax
            g(ix, iy2, iz) = 0.0d0; phi(ix, iy2, iz) = vmax
            g(ix, iy, iz2) = 0.0d0; phi(ix, iy, iz2) = vmax
        enddo
    enddo
    return
end subroutine initial

!*******************************************************************************
! Central difference to calculate gradient
!*******************************************************************************
!subroutine gradient_g(nx, ny, nz, g, gx, gy, gz, dx)
!    implicit real*8 (a-h, o-z)
!    dimension g(nx, ny, nz), gx(nx, ny, nz), gy(nx, ny, nz), gz(nx, ny, nz)
!    dimension w(-1:1)
    
!    w(-1) = -5.d-1/dx; w(0) = 0.0d0; w(1) = 5.d-1/dx
    
!    do iz = 2, nz - 1
!        do iy = 2, ny - 1
!            do ix = 2, nx - 1
!                gx(ix, iy, iz) = w(-1) * g(ix - 1, iy, iz) + w(1) * g(ix + 1, iy, iz)
!                gy(ix, iy, iz) = w(-1) * g(ix, iy-1, iz) + w(1) * g(ix, iy + 1, iz)
!                gz(ix, iy, iz) = w(-1) * g(ix, iy, iz-1) + w(1) * g(ix, iy, iz + 1)
!            enddo
!        enddo
!    enddo
!    return
!end subroutine gradient_g

!*******************************************************************************
! Upwinding scheme, neumann bc for phi
!*******************************************************************************


