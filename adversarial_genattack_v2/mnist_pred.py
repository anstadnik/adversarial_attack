import tensorflow.compat.v1 as tf
from setup_mnist import MNIST, MNISTModel
import utils
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'
tf.disable_v2_behavior()

def pred(img):

    with tf.Session() as sess:
        model = MNISTModel('models/mnist', sess, use_log=True)
        image_dim = 28
        image_channels = 1
        num_labels = 10

        test_in = tf.placeholder(
            tf.float32, (1, image_dim, image_dim, image_channels), 'x')
        # test_pred = tf.argmax(model.predict(test_in), axis=1)
        test_pred = model.predict(test_in)

        orig_pred = sess.run(test_pred, feed_dict={
                             test_in: [img]})[0]


        px.imshow(img.reshape((28, 28))).show()
        print(orig_pred)

def main():
    dataset = MNIST()
    inputs, targets, reals = utils.generate_data(dataset, 10)
    pred(inputs[0])

if __name__ == "__main__":
    main()
