const model = tf.sequential();

/**
 * hidden layer has:
 *  Four units(nodes)
 * inputShape : The first layer in a Sequential model must get an `inputShape` or `batchInputShape` argument.
 */

// dense is a "fully connected layer".
const hidden = tf.layers.dense({
    units: 4,
    inputShape: [2], // 2 for inputShape.
    activation: 'sigmoid'
});

/**
 * output layer has:
 *  Four units(nodes)
 */
const output = tf.layers.dense({
    units: 3,
    // here the input shape is "inferred from the previous layer"
    activation: 'sigmoid'
});

model.add(hidden);
model.add(output);

// An optimizer using gradient descent.
const sgdOpt = tf.train.sgd(0.1);

// I'm done configuring the model so compile it.
model.compile({
    optimizer: sgdOpt,
    loss: tf.losses.meanSquaredError
});

const xs = tf.tensor2d([
    [0.25, 0.92],
    [0.12, 0.3],
    [0.1, 0.22]]);

const ys = tf.tensor2d([
        [0.20, 0.02, 0.02],
        [0.12, 0.31, 0.22],
        [0.08, 0.91, 0.10]]);

model.fit(xs, ys).then(response => console.log(response.history))

/* const inputs = tf.tensor2d([
    [0.25, 0.92],
    [0.12, 0.3],
    [0.4, 0.74],
    [0.1, 0.22]]);
let outputs = model.predict(inputs);
outputs.print(); */
