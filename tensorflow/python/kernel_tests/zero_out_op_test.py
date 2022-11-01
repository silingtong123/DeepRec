import tensorflow as tf
from tensorflow.python.platform import test
class ZeroOutTest(test.TestCase):
  def testZeroOut(self):
    with self.test_session():
      result = tf.user_ops.zero_out([5, 4, 3, 2, 1])
      print(result.eval())
      self.assertAllEqual(result.eval(), [5, 0, 0, 0, 0])
