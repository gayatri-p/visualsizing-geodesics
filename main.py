from manim import *
from math import *
from solver import *

def trajectory(r0 = 3, E=1, Lz=5, M=1, epsilon=0):
    schwarz = Schwarzschild(M = M)

    # Initial Conditions
    t0 = 0
    r0 = r0
    theta0 = np.pi/2
    phi0 = 0

    # p0 = 0 # time
    # p1 = 0 # r
    p2 = 0 # theta
    # p3 = 0 # phi
    
    n_rays = 1 # number of light rays
    N = 100 # number of simulation points

    lines = []
    for epsilon in [0, -1]:
        p0, p1, p3 = schwarz.compute_4momentum(r0, E, Lz, epsilon)
        y0 = [t0, r0, theta0, phi0, p0, p1, p2, p3]

        # Solve the equations of motion for a specified number of steps with a specified initial step size
        if epsilon == -1:
            tau, sol = schwarz.solve(y0, N, 1e-5)
        else:
            tau, sol = schwarz.solve(y0, N, 1e-5)

        t, r, theta, phi = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3]

        x = r*np.cos(phi)*np.sin(theta)
        y = r*np.sin(phi)*np.sin(theta)
        lines.append(np.array([x, y, np.zeros(N)]).T)
    
    return lines, schwarz.r_EH

def circle(r):
    theta = np.linspace(-2*np.pi, 2*np.pi, 500)
    return np.array([r*np.cos(theta), r*np.sin(theta), np.zeros(500)]).T

class BlackHole(Scene):
    def construct(self):
        self.camera.background_color = ManimColor('#191919')
        axes = Axes(
            x_range=(-10, 10),
            y_range=(-10, 10),
            x_length=12, y_length=12
        )

        # Initial Conditions
        r0 = 10
        E = 1.01
        Lz = 4.5
        M = 1
        epsilon = 0
        all_photons, r_EH = trajectory(r0, E, Lz, M, epsilon)
        self.add(Tex(r'$E =$' + f' {E},' +r' $M_{BH}=$'+f' {M},' +r' $L_z=$' + f' {Lz:.2f}', font_size=30).to_edge(DR).set_color(WHITE))
        self.add(MarkupText('Trajectory of a', font_size=30).to_edge(UL).set_color(WHITE))
        self.add(MarkupText(f'<span fgcolor="{YELLOW}">Photon</span> and a', font_size=30).next_to(self.mobjects[-1],DOWN))
        self.add(MarkupText(f'<span fgcolor="{BLUE}">Massive Particle</span>', font_size=30).next_to(self.mobjects[-1],DOWN))
        
        curves = VGroup()
        colors = [YELLOW, BLUE]
        for i, points in enumerate(all_photons):
            curve = VMobject().set_points_as_corners(axes.c2p(points))
            curve.set_stroke(colors[i], 3)
            curves.add(curve)

        black_hole_core = VMobject().set_points_as_corners(axes.c2p(circle(r_EH)))
        black_hole_core.set_stroke(
                opacity=0
        )
        black_hole_core.set_fill(BLACK, opacity=1.0)
        self.add(black_hole_core)
        black_hole_core.z_index = 1

        photosphere = VMobject().set_points_as_corners(axes.c2p(circle(3*M)))
        photosphere.set_stroke(
                color=RED, 
                width=2,
                opacity=0.6 
        )
        dashed_photosphere = DashedVMobject(photosphere)
        self.add(dashed_photosphere)

        # Animate the light ray being traced
        self.play(
            *(
                Create(curve, rate_func=linear)
                for curve in curves
            ),
            run_time=2,
        )
        self.wait(1)