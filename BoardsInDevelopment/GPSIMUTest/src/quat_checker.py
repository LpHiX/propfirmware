

import unittest


class QuaternionTestWrapper:
    """Wrapper class to test the quaternion multiply function"""
    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions in scalar-last format [x,y,z,w]"""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return [x, y, z, w]

class TestQuaternionMultiply(unittest.TestCase):
    
    def setUp(self):
        self.quat_wrapper = QuaternionTestWrapper()
    
    def test_identity_quaternion(self):
        """Test multiplication with identity quaternion [0,0,0,1]"""
        identity = [0, 0, 0, 1]
        q = [0.5, -0.5, 0.5, 0.5]
        
        result = self.quat_wrapper.quaternion_multiply(q, identity)
        self.assertAlmostEqual(result[0], q[0])
        self.assertAlmostEqual(result[1], q[1])
        self.assertAlmostEqual(result[2], q[2])
        self.assertAlmostEqual(result[3], q[3])
        
        result = self.quat_wrapper.quaternion_multiply(identity, q)
        self.assertAlmostEqual(result[0], q[0])
        self.assertAlmostEqual(result[1], q[1])
        self.assertAlmostEqual(result[2], q[2])
        self.assertAlmostEqual(result[3], q[3])
    
    def test_known_multiplication(self):
        """Test with known quaternion multiplications"""
        q1 = [1, 0, 0, 0]  # [x,y,z,w] = [1,0,0,0]
        q2 = [0, 1, 0, 0]  # [x,y,z,w] = [0,1,0,0]
        
        # Known result: q1 * q2 = [0,0,1,0]
        expected = [0, 0, 1, 0]
        result = self.quat_wrapper.quaternion_multiply(q1, q2)
        
        for i in range(4):
            self.assertAlmostEqual(result[i], expected[i])
    
    def test_non_commutativity(self):
        """Test that quaternion multiplication is not commutative"""
        q1 = [1, 2, 3, 4]
        q2 = [5, 6, 7, 8]
        
        result1 = self.quat_wrapper.quaternion_multiply(q1, q2)
        result2 = self.quat_wrapper.quaternion_multiply(q2, q1)
        
        # Results should be different
        self.assertNotEqual(result1, result2)
    
    def test_unit_quaternion_magnitude(self):
        """Test that multiplication of unit quaternions preserves magnitude"""
        # Unit quaternions
        q1 = [0, 0, 0, 1]
        q2 = [0.5, 0.5, 0.5, 0.5]
        
        # Normalize q2
        norm = (q2[0]**2 + q2[1]**2 + q2[2]**2 + q2[3]**2)**0.5
        q2 = [q2[i]/norm for i in range(4)]
        
        result = self.quat_wrapper.quaternion_multiply(q1, q2)
        result_norm = (result[0]**2 + result[1]**2 + result[2]**2 + result[3]**2)**0.5
        
        self.assertAlmostEqual(result_norm, 1.0)
    
    def test_zero_quaternion(self):
        """Test multiplication with zero quaternion"""
        zero_quat = [0, 0, 0, 0]
        q = [1, 2, 3, 4]
        
        result = self.quat_wrapper.quaternion_multiply(q, zero_quat)
        expected = [0, 0, 0, 0]
        
        for i in range(4):
            self.assertAlmostEqual(result[i], expected[i])

if __name__ == "__main__":
    unittest.main()