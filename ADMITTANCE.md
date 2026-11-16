# Admittance control

## 1. Turn a push into joint torque (Lect. 5)

The point Jacobian $J_c$ gives a first-order approximation of how the clicked
point moves when the joint angles $q$ change. Equating virtual work at the point
and at the joints gives the torque from the applied force $F$.

$$
\boxed{\Delta p_c \approx J_c(q)\Delta q,\qquad \tau_{\rm ext}=J_c(q)^\top F}
$$

PyBullet applies the force at the selected point and recomputes $J_c$ as the
arm moves. The same force can have different leverage at different poses.
[Lynch and Park, Modern Robotics §5.2](https://modernrobotics.northwestern.edu/nu-gm-book-resource/5-2-statics-of-open-chains/)

## 2. Turn joint torque into motion (Lect. 6–7)

While you push, pick-and-place pauses at the joint target $q_*$. The controller
uses a virtual inertia $m=0.30\ \mathrm{kg\,m^2}$ and damping
$d=6\ \mathrm{N\,m\,s/rad}$ to compute the commanded offset $e$ and rate $v$.

$$
\boxed{m\dot v+d v=\tau_{\rm ext},\qquad \dot e=v,\qquad q_{\rm cmd}=q_*+e}
$$

**Lect. 6:** For one virtual joint, the undamped double integrator has
$A=\begin{bmatrix}0&1\\0&0\end{bmatrix}$ and $B=[0,1/m]^\top$. The
controllability matrix $[B,AB]$ has rank two. On the robot, $J_c^\top F$
determines the joint directions the push can excite.

**Lect. 7:** The feedback term $-dv$ gives the velocity a pole at
$-d/m=-20\ \mathrm{s}^{-1}$. The zero position pole integrates velocity.
Under constant torque the rate approaches $\tau_{\rm ext}/d$, so a sustained
push keeps adding displacement.
[Keemink et al., Admittance control](https://doi.org/10.1177/0278364918768950)

### How the wrist follows

At each tick, the controller projects the previous rate into the contact
Jacobian's row space, $P=J_c^+J_c$, before stepping the mass–damper equation.
The projection $N=I-P$ gives $J_cN\approx 0$. This lets the wrist correct the
gripper's orientation with little effect on the clicked point's velocity.

$$
\begin{aligned}
v_{\rm push} &= P v + \frac{\Delta t}{m}(J_c^\top F-dPv),\\
A &= J_\omega N,\qquad
v_{\rm follow}=A^\top(AA^\top+\lambda^2I)^{-1}(4r-J_\omega v_{\rm push}),\\
e_{\rm next} &= e+\Delta t\,(v_{\rm push}+v_{\rm follow}).
\end{aligned}
$$

Here $J_\omega$ is the tool's angular Jacobian and $r$ is the orientation error
from the start of the push. The weight $\lambda=0.10$ regularizes the wrist solve.
The update runs at 60 Hz and applies joint speed, position, and tracking limits.
[Sciavicco and Siciliano, Task priority for redundant manipulators](https://doi.org/10.1177/027836499101000409)

## 3. Return after release (Lect. 8–10)

Use the **double integrator** from Lect. 8 and 10 with the **finite-horizon cost**
from Lect. 9. Starting from the release offset $e_0$ and rate $v_0$, choose the
acceleration $u$ that minimizes squared acceleration and returns to the
waypoint at rest.

$$
\boxed{\min_u \frac12\int_0^T \lVert u(t)\rVert^2\,dt,\qquad
\ddot e=u,\quad (e,\dot e)(0)=(e_0,v_0),\quad (e,\dot e)(T)=(0,0)}
$$

The optimality condition is $e^{(4)}=0$. Applying the four endpoint conditions
gives the cubic below, where $s=t/T$.

$$
\boxed{e(t)=(1-3s^2+2s^3)e_0+T(s-2s^2+s^3)v_0}
$$

The return takes at least 0.65 s and gets longer with displacement. Near a joint
limit, the planner adjusts the cubic's initial rate to keep the curve within
the joint bounds. The task resumes when the offset reaches zero. The return
assumes clearance around the arm, which is easiest to demonstrate while carrying.
