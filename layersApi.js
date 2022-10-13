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
    [0, 0.1],
    [0.2, 0.3],
    [0.4, 0.5]]);

const ys = tf.tensor2d([
        [0, 0.2, 0.3],
        [0.4, 0.6, 0.7],
        [0.8, 0.10, 0.11]]);

train().then(() => {
    let outputs = model.predict(xs);
    outputs.print(); 
});

async function train(){
    for(i = 0; i < 1000; i++){
    const config = {
        shuffle: true,
        epochs: 10
    }
    const response = await model.fit(xs, ys, config);
    console.log(response.history.loss[0]);
    }
}

/* const inputs = tf.tensor2d([
    [0.25, 0.92],
    [0.12, 0.3],
    [0.4, 0.74],
    [0.1, 0.22]]);
let outputs = model.predict(inputs);
outputs.print(); */
