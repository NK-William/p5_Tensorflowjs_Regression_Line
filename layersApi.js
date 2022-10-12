const model = tf.sequential();

/**
 * hidden layer has:
 *  Four units(nodes)
 * inputShape : The first layer in a Sequential model must get an `inputShape` or `batchInputShape` argument.
 */
const hidden = tf.layers.dense({
    units: 4,
    inputShape: [2], // 2 for input nodes.
    activation: 'sigmoid'
});

/**
 * output layer has:
 *  Four units(nodes)
 */
const output = tf.layers.dense({
    units: 3,
    activation: 'sigmoid'
});

model.add(hidden);
model.add(output);

const sgdOpt = tf.train.sgd(0.1);
model.compile({
    optimizer: sgdOpt,
    loss: tf.losses.cosineDistance
});