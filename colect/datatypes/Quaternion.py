import numpy as np

from numpy.typing import ArrayLike


class Quaternion:
    """A class that implements a representation of quaternions, as well as operations on them.

    Attributes
    ----------
    v : float
        The scalar part of the quaternion.
    u : array-like of shape (3,)
        The complex part of the quaternion.
    """

    def __init__(self, v: float, u: ArrayLike) -> None:
        self.v = v
        self.u = u
        if self.v is not None: self.normalize()

    @classmethod
    def from_rotation_vector(cls, rotation_vector: ArrayLike) -> 'Quaternion':
        """Creates a quaternion from a rotation vector.

        Parameters
        ----------
        rotation_vector : array-like of shape (3,)
            The rotation vector, i.e. an axis-angle representation with the angle premultipled to the
            axis.

        Returns
        -------
        quaternion
            The quaternion representation of the rotation vector.
        """
        angle = np.linalg.norm(rotation_vector)
        axis = rotation_vector/angle
        v = np.abs(np.cos(angle/2))
        u = axis*np.sin(angle/2)
        return cls(v, u)
    
    def as_rotation_vector(self) -> ArrayLike:
        """Converts the quaternion to a rotation vector.

        Returns
        -------
        array-like of shape (3,)
            The rotation vector.
        """
        angle = 2*np.arccos(self.v)
        if np.abs(angle) > 1e-8:
            axis = self.u / np.sin(angle/2)
        else:
            axis = np.zeros(3)
        return angle*axis

    @classmethod
    def from_array(cls, array: ArrayLike) -> 'Quaternion':
        """Creates a quaternion from an array.

        Parameters
        ----------
        array : array-like of shape (4,)
            Array holding the quaternion components. The scalar part should be the first element.

        Returns
        -------
        quaternion
            The corresponding quaternion
        """
        return cls(array[0], array[1:])
    
    def as_array(self) -> ArrayLike:
        """Returns the quaternion's components as an array, with the scalar component first.

        Returns
        -------
        array-like of shape (4,)
            The array representing the quaternion.
        """
        return np.concatenate(([self.v], self.u))

    @classmethod
    def from_rotation_matrix(cls, R: ArrayLike) -> 'Quaternion':
        """Creates a quaternion from a rotation matrix.

        Parameters
        ----------
        array : array-like of shape (3,3)
            The rotation matrix.

        Returns
        -------
        quaternion
            The corresponding quaternion
        """
        trace = R[0,0] + R[1,1] + R[2,2]
        if trace > 0:
            S = np.sqrt(trace + 1) * 2
            v = 0.25 * S
            u = np.array([(R[2,1] - R[1,2]) / S, (R[0,2] - R[2,0]) / S, (R[1,0] - R[0,1]) / S])
        elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            S = np.sqrt(1 + R[0,0] - R[1,1] - R[2,2]) * 2
            v =(R[2,1] - R[1,2]) / S
            u = np.array([0.25 * S, (R[0,1] - R[1,0]) / S, (R[0,2] - R[2,0]) / S])
        elif R[1,1] > R[2,2]:
            S = np.sqrt(1 + R[1,1] - R[0,0] - R[2,2]) * 2
            v = (R[0,2] - R[2,0]) / S
            u = np.array([(R[0,1] - R[1,0]) / S, 0.25 * S, (R[1,2] - R[2,1]) / S])
        else:
            S = np.sqrt(1 + R[2,2] - R[0,0] - R[1,1]) * 2
            v = (R[1,0] - R[0,1]) / S
            u = np.array([(R[0,2] - R[2,0]) / S, (R[1,2] - R[2,1]) / S, 0.25 * S])
        return cls(v, u)

    def as_rotation_matrix(self) -> np.array:
        """Return the rotation matrix representation of self.
        
        Returns
        -------
        ArrayLike
            The rotation matrix corresponding to self.
        """
        w = self.v
        x, y, z = self.u
        out = np.array([[1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                        [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
                        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2 ]])
        return out

    @classmethod
    def exp(cls, w: ArrayLike) -> 'Quaternion':
        """Project an R^3 vector to quaternion space.

        Parameters
        ----------
        w : array-like of shape (3,)
            The vector to project.

        Returns
        -------
        quaternion
            The result of the projection.
        """
        norm = np.linalg.norm(w)
        v = 1
        u = np.zeros(3)
        if not np.allclose(w, np.zeros_like(w)):
            v = np.cos(norm)
            u = np.sin(norm)*w/norm
        return cls(v, u)

    def log(self) -> ArrayLike:
        """Projects the quaternion to Euclidean space.

        Returns
        -------
        array-like of shape (3,)
            The projection in R^3 of self.
        """
        norm = np.linalg.norm(self.as_array()) + np.finfo(np.float64).tiny 
        u = np.sign(self.v)*self.u/norm
        v = np.sign(self.v)*self.v/norm
        if not np.allclose(u, np.zeros_like(u)):
            return np.arccos(v)*u/np.linalg.norm(u)
        else:
            return np.zeros_like(u)

    def abs(self) -> ArrayLike:
        """Compute the absolute value of the quaternion's components.

        Returns
        -------
        array-like of shape (4,)
            The vector of absolute values of self's components.
        """
        return np.concatenate(([np.abs(self.v)], np.abs(self.u)))

    def normalize(self) -> 'Quaternion':
        """Normalizes self.

        Returns
        -------
        quaternion
            The result of the normalization.
        """
        norm = np.linalg.norm(self.as_array()) + np.finfo(np.float64).tiny 
        self.u = self.u/norm
        self.v = self.v/norm
        return self

    def derivative(self, wh: ArrayLike, qa: 'Quaternion', dt: float) -> ArrayLike:
        """Computes the Euclidean projection of the derivative of self.

        Parameters
        ----------
        wh : ArrayLike
            The angular velocity associated to self.
        qa : quaternion
            The auxiliary quaternion.
        dt : float
            The timestep for the derivative.

        Returns
        -------
        ArrayLike
            The derivative of the Euclidean projection of self.
        """
        xi1 = ((self.exp(wh*dt/2)*self)*~qa).log()
        xi2 = (self*~qa).log()
        return (xi1 - xi2)/dt

    def __add__(self, q: 'Quaternion') -> 'Quaternion':
        """Overload the '+' operator to handle quaternion-quaternion sum.

        Parameters
        ----------
        q : quaternion
            The second quaternion

        Returns
        -------
        quaternion
            The result of the addition self+q.
        """
        return Quaternion(self.v+q.v, self.u+q.u)

    def __sub__(self, q: 'Quaternion') -> 'Quaternion':
        """Overload the '+' operator to handle quaternion-quaternion subtraction.

        Parameters
        ----------
        q : quaternion
            The second quaternion

        Returns
        -------
        quaternion
            The result of the subtraction self-q.
        """
        return Quaternion(self.v-q.v, self.u-q.u)

    def __mul__(self, q: 'Quaternion') -> 'Quaternion':
        """Overload the '*' operator to handle quaternion-quaternion multiplication or 
        quaternion-scalar multiplication.

        Parameters
        ----------
        q : quaternion or float
            The quantity to multiply the quaternion by.

        Returns
        -------
        quaternion
            The result of the quaternion multiplication self*q.
        """
        if isinstance(q, Quaternion):
            u = self.v*q.u + q.v*self.u + np.cross(self.u, q.u)
            v = self.v*q.v - self.u.T@q.u
        else:
            u = self.u*q
            v = self.v*q
        return Quaternion(v, u)

    def __truediv__(self, q: float) -> 'Quaternion':
        """Overload the '/' operator to handle quaternion-scalar division.

        Parameters
        ----------
        q : float
            The quantity to divide the quaternion by.

        Returns
        -------
        quaternion
            The result of the division self/q
        """
        return Quaternion(self.v/q, self.u/q)

    def __invert__(self) -> 'Quaternion':
        """Overload the '~' operator to handle quaternion conjugation.

        Returns
        -------
        quaternion
            The conjugate quaternion of self.
        """
        return Quaternion(self.v, -self.u)

    def __neg__(self) -> 'Quaternion':
        """Overload the '-' unary operator to handle flipping the sign of a quaternion.

        Returns
        -------
        quaternion
            Self with opposite sign.
        """
        return Quaternion(-self.v, -self.u)

    def __pos__(self) -> 'Quaternion':
        """Overload the '+' unary operator.

        Returns
        -------
        quaternion
            A copy of self.
        """
        return Quaternion(self.v, self.u)

    def __getitem__(self, index: int) -> float:
        """Overload subscription.

        Parameters
        ----------
        index : int
            The index of the desired quaternion component. The scalar part is at index 0, the complex
            part at indexes 1:3.

        Returns
        -------
        float
            The requested quaternion component.
        """
        if index == 0:
            return self.v
        else:
            return self.u[index-1]

    def __setitem__(self, index: int, value: float) -> None:
        """Overload subscription.

        Parameters
        ----------
        index : int
            The index of the desired quaternion component. The scalar part is at index 0, the complex
            part at indexes 1:3.
        value : float
            The value to assign to the component.
        """
        if index == 0:
            self.v = value
        else:
            self.u[index-1] = value

    def __str__(self) -> str:
<<<<<<< HEAD
        """String representation of self"""
        return f"{self.u[0]}, {self.u[1]}, {self.u[2]}, {self.v}"
=======
        """Class string representation.
        """
        return f"{self.as_array()}"

>>>>>>> a5c307bba7455dc5ad407b051db178a01091485c
