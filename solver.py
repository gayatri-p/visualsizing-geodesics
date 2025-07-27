import numpy as np 

class Schwarzschild:
    def __init__(self, M):
        self.M = M
        self.r_EH = 2*self.M
    
    def g_tt(self, r, theta):
        return -(1 - 2*self.M/r)
    
    def g_rr(self, r, theta):
        return 1/(1 - 2*self.M/r)

    def g_thth(self, r, theta):
        return r**2

    def g_phph(self, r, theta):
        return r**2*np.sin(theta)**2    
    
    def geodesic_eq_t(self, y):
        t, r, theta, phi, p0, p1, p2, p3 = y  
        
        return -2*self.M*p1*p0/(r*(r-2*self.M))

    def geodesic_eq_r(self, y):
        t, r, theta, phi, p0, p1, p2, p3 = y  
        sin = np.sin(theta)
        M = self.M
        ch0 = M*(r-2*M)/r**3
        ch1 = -M/(r*(r-2*M))
        ch2 = -(r-2*M)
        ch3 = -(r-2*M)*sin**2
        return -(ch0*p0**2 + ch1*p1**2 + ch2*p2**2 + ch3*p3**2) 
    
    def geodesic_eq_theta(self, y):
        t, r, theta, phi, p0, p1, p2, p3 = y  
        sin = np.sin(theta)
        cos = np.cos(theta)
        M = self.M
        return -(2*p1*p2/r - sin*cos*p3**2)
    
    def geodesic_eq_phi(self, y):
        t, r, theta, phi, p0, p1, p2, p3 = y  
        sin = np.sin(theta)
        cos = np.cos(theta)
        return -2*(p1*p3/r - cos*p2*p3/sin)
    
    def compute_4momentum(self, r0, E, Lz, epsilon = 0):
        p0 = E/(1 - 2*self.M/r0)
        p3 = Lz/r0**2
        try:
            # negative solution spirals inward
            p1 = -np.sqrt(E**2 - (1-2*self.M/r0)*(Lz**2/r0**2 - epsilon))
        except:
            print('This set of inital conditions is forbidden.')
            p1 = 0
        return p0, p1, p3

    def geodesic_eq(self, y):
        """
        Return 8 differential equations which solve for the geodesic.
        """
        t, r, theta, phi, p0, p1, p2, p3 = y

        # Initializing the derivatives of the coordinates and four-momentum of our particle
        derivatives = np.zeros_like(y)

        derivatives[0] = p0
        derivatives[1] = p1
        derivatives[2] = p2
        derivatives[3] = p3
        derivatives[4] = self.geodesic_eq_t(y)
        derivatives[5] = self.geodesic_eq_r(y)
        derivatives[6] = self.geodesic_eq_theta(y)
        derivatives[7] = self.geodesic_eq_phi(y)
        
        return derivatives
    
    def RKF45(self, y, h, tol = 1e-12):
        abs_error = np.inf

        while abs_error > tol:
            k1 = self.geodesic_eq(y)
            k2 = self.geodesic_eq(y +       1/4*k1*h)
            k3 = self.geodesic_eq(y +      3/32*k1*h +      9/32*k2*h)
            k4 = self.geodesic_eq(y + 1932/2197*k1*h - 7200/2197*k2*h + 7296/2197*k3*h)
            k5 = self.geodesic_eq(y +   439/216*k1*h -         8*k2*h +  3680/513*k3*h -  845/4104*k4*h)
            k6 = self.geodesic_eq(y -      8/27*k1*h +         2*k2*h - 3544/2565*k3*h + 1859/4104*k4*h - 11/40*k5*h)

            new_y = y + 16/135*k1*h + 6656/12825*k3*h + 28561/56430*k4*h - 9/50*k5*h + 2/55*k6*h
            error = -1/360*k1*h + 128/4275*k3*h + 2197/75240*k4*h - 1/50*k5*h - 2/55*k6*h
            abs_error = np.sqrt(np.sum(error**2))
            new_h = 0.9*h*(tol/abs_error)**(1/5)
            h = new_h

        return new_y, h
    
    def solve(self, y0, n, h0):
        affine_parameter = np.zeros(n)

        # Solution Array
        y = np.zeros((n, len(y0)))
        y[0] = y0
        h = h0 # Initial step size

        # Performing numerical integration
        for i in range(n - 1):
            y_next, h = self.RKF45(y[i], h)

            y[i + 1] = y_next
            affine_parameter[i + 1] = affine_parameter[i] + h

        return affine_parameter, y